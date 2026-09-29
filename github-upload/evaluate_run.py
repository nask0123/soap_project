"""HELDOUT_EVAL_V1: fixed held-out evaluation, no training or threshold tuning."""
import contextlib
import hashlib
import json
import subprocess
import sys
import traceback
import argparse
from datetime import datetime, timezone
from pathlib import Path

parser = argparse.ArgumentParser(description='Evaluate a saved full run on the fixed held-out 100 pairs.')
parser.add_argument('source', type=Path, help='Full run directory')
parser.add_argument('--output-root', type=Path)
args = parser.parse_args()
SOURCE = args.source.resolve()
ROOT = args.output_root.resolve() if args.output_root else SOURCE.parent
ROOT.mkdir(parents=True, exist_ok=True)
OUT = ROOT / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-heldout-eval')
OUT.mkdir(parents=True)
report = {'status': 'STARTED', 'output': str(OUT), 'source': str(SOURCE)}

def log_command(name, args):
    with (OUT / name).open('w') as f:
        f.write(f'START {datetime.now(timezone.utc).isoformat()} {args!r}\n')
        f.flush()
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(ROOT))
        for line in p.stdout:
            print(line, end='', flush=True)
            f.write(datetime.now(timezone.utc).isoformat()+' '+line)
        rc = p.wait()
        f.write(f'END {datetime.now(timezone.utc).isoformat()} exit_code={rc}\n')
    return rc

def main():
    # Separate process: validate save/reload before evaluating the full adapter.
    rc = log_command('full-reload.log', [sys.executable, '-u', str(Path(__file__).with_name('verify_adapter.py')), str(SOURCE), '--output-root', str(ROOT)])
    assert rc == 0, 'Full adapter reload verification failed; do not score yet'

    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    adapter = SOURCE / 'adapter'
    cfg = json.loads((adapter / 'adapter_config.json').read_text())
    tok = AutoTokenizer.from_pretrained(adapter)
    base_path = cfg['base_model_name_or_path']
    if not Path(base_path).is_dir():
        from huggingface_hub import snapshot_download
        revision = json.loads((SOURCE/'smoke-report.json').read_text())['model_revision']
        assert revision == '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
        base_path = snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', revision=revision,
            allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
    model = AutoModelForCausalLM.from_pretrained(base_path, dtype=torch.float16, device_map='cuda')
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    test_path = SOURCE / 'test-conversational.jsonl'
    rows = [json.loads(s) for s in test_path.read_text().splitlines()]
    assert len(rows) == 100
    report['test_sha256'] = hashlib.sha256(test_path.read_bytes()).hexdigest()
    report['metric'] = 'Preference ranking accuracy: chosen completion sum log-probability > rejected; ties count as 0. Not human-judged generation quality.'
    report['precision'] = 'FP16 base and CUDA FP16 autocast for BOTH baseline and tuned; log_softmax in FP32; no truncation'

    def score(row, disabled):
        prompt = tok.apply_chat_template(row['prompt'], tokenize=True, return_dict=False, add_generation_prompt=True)
        sequences = [tok.apply_chat_template(row['prompt']+row[k], tokenize=True, return_dict=False) for k in ('chosen','rejected')]
        assert all(s[:len(prompt)] == prompt and len(s) <= 512 for s in sequences)
        width = max(map(len, sequences))
        ids = torch.full((2,width), tok.pad_token_id, dtype=torch.long, device='cuda')
        attention = torch.zeros_like(ids)
        mask = torch.zeros((2,width-1), dtype=torch.bool, device='cuda')
        for i,s in enumerate(sequences):
            ids[i,:len(s)] = torch.tensor(s, device='cuda')
            attention[i,:len(s)] = 1
            mask[i,len(prompt)-1:len(s)-1] = True
            assert tok.eos_token_id in s[len(prompt):]
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16), (model.disable_adapter() if disabled else contextlib.nullcontext()):
            logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits
        selected = logits[:,:-1].float().log_softmax(-1).gather(-1, ids[:,1:,None]).squeeze(-1)
        assert torch.isfinite(selected[mask]).all()
        totals = (selected*mask).sum(-1)
        lengths = mask.sum(-1)
        return totals.cpu().tolist(), lengths.cpu().tolist()

    results = []
    raw_path = OUT / 'heldout-rows.jsonl'
    with raw_path.open('w') as f:
        for i,row in enumerate(rows):
            base, lengths = score(row, True)
            tuned, lengths2 = score(row, False)
            assert lengths == lengths2
            item = {'index': i, 'base_sum_logp': base, 'tuned_sum_logp': tuned, 'completion_lengths': lengths,
                'base_correct': base[0] > base[1], 'tuned_correct': tuned[0] > tuned[1],
                'base_mean_correct': base[0]/lengths[0] > base[1]/lengths[1],
                'tuned_mean_correct': tuned[0]/lengths[0] > tuned[1]/lengths[1],
                'preference_margin_change': (tuned[0]-tuned[1])-(base[0]-base[1])}
            results.append(item)
            f.write(json.dumps(item)+'\n')
            f.flush()
            if (i+1)%10 == 0:
                print(f'HELDOUT: {i+1}/100 pairs evaluated', flush=True)
    base_acc = np.array([r['base_correct'] for r in results], dtype=float)
    tuned_acc = np.array([r['tuned_correct'] for r in results], dtype=float)
    delta = tuned_acc-base_acc
    rng = np.random.default_rng(42)
    boots = delta[rng.integers(0,100,size=(10000,100))].mean(axis=1)
    report.update({
        'base_accuracy': float(base_acc.mean()), 'tuned_accuracy': float(tuned_acc.mean()),
        'accuracy_delta': float(delta.mean()),
        'paired_bootstrap_95ci': np.quantile(boots,[0.025,0.975]).tolist(),
        'wrong_to_right': int((delta>0).sum()), 'right_to_wrong': int((delta<0).sum()),
        'base_length_normalized_accuracy': float(np.mean([r['base_mean_correct'] for r in results])),
        'tuned_length_normalized_accuracy': float(np.mean([r['tuned_mean_correct'] for r in results])),
        'mean_preference_margin_change': float(np.mean([r['preference_margin_change'] for r in results])),
        'fraction_positive_margin_change': float(np.mean([r['preference_margin_change']>0 for r in results])),
    })
    report['length_strata'] = {}
    for name, pred in [('chosen_longer',lambda n:n[0]>n[1]), ('chosen_not_longer',lambda n:n[0]<=n[1])]:
        subset = [r for r in results if pred(r['completion_lengths'])]
        report['length_strata'][name] = {'n':len(subset),
            'base_accuracy': float(np.mean([r['base_correct'] for r in subset])) if subset else None,
            'tuned_accuracy': float(np.mean([r['tuned_correct'] for r in subset])) if subset else None}

    # No invented forgetting scores: empty benchmarks explicitly exposes the missing evidence.
    evidence = {'task': {'mode':'metric', 'base':report['base_accuracy'], 'tuned':report['tuned_accuracy']}, 'benchmarks': {}}
    evpath = OUT / 'ship-evidence.json'
    evpath.write_text(json.dumps(evidence, indent=2))
    report['ship_evidence_limitations'] = 'No independent general-capability/forgetting benchmark measured; empty benchmarks is intentional missing evidence, not zero regression.'
    report['ship_exit_code'] = log_command('soup-ship.log', [str(Path(sys.executable).with_name('soup')), 'ship', '--evidence', str(evpath), '--output', str(OUT/'ship-verdict.json')])
    assert report['ship_exit_code'] in (0,2), 'Inspect soup ship runtime/usage failure'
    report['status'] = 'EVALUATION_COMPLETE'
    report['limitations'] = '100 length-filtered general preference pairs; one seed; no Russian-support evaluation, no broad forgetting evaluation, no gradient parity. Bootstrap covers pair sampling only, not training randomness.'

if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        report['status'] = 'FAILED'
        report['error'] = repr(exc)
        (OUT/'traceback.txt').write_text(traceback.format_exc())
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        (OUT/'evaluation-report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2),flush=True)
