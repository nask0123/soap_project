"""Audited Soup DPO smoke run with AMP_SKIP_AUDIT_V2. Run in Colab's Python 3.12 venv."""
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description='Instrumented Soup DPO: smoke or one full epoch on T4.')
parser.add_argument('--full', action='store_true', help='Fresh one-epoch run rather than bounded smoke diagnostic')
parser.add_argument('--data-dir', type=Path, default=REPO / 'data')
parser.add_argument('--config', type=Path, default=REPO / 'config/soup.yaml')
parser.add_argument('--output-root', type=Path, default=REPO / 'runs')
parser.add_argument('--model-cache', type=Path, default=Path.home() / '.cache/soup-takehome/models')
args = parser.parse_args()
ROOT = args.output_root.resolve()
ROOT.mkdir(parents=True, exist_ok=True)
FULL_RUN = args.full  # FULL_DPO_V1: fresh one-epoch training, not smoke resume.
DATA = args.data_dir.resolve()
DRAFT = args.config.resolve()
OUT = ROOT / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + ('-full-dpo' if FULL_RUN else '-smoke'))
OUT.mkdir(parents=True)
REPORT = {'status': 'STARTED', 'output': str(OUT), 'scope': 'at most eight attempted steps; require two actual updates; not a ship verdict'}
if FULL_RUN:
    REPORT['scope'] = 'Fresh adapter; one epoch over 500 pairs; no held-out quality claim yet'
monitor = None
monitor_file = None

def save_report():
    (OUT / 'smoke-report.json').write_text(json.dumps(REPORT, indent=2, default=str))

def read_rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]

def main():
    global monitor, monitor_file
    import torch
    import yaml
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, TrainerCallback
    from soup_cli.config.loader import load_config
    from soup_cli.data.loader import load_dataset
    from soup_cli.trainer.dpo import DPOTrainerWrapper
    from soup_cli.utils.mixed_precision import align_trainable_dtype_for_fp16
    from safetensors.torch import save_file

    assert torch.cuda.is_available(), 'CUDA unavailable'
    assert 'T4' in torch.cuda.get_device_name(0), 'This run requires T4'
    REPORT['gpu'] = torch.cuda.get_device_name(0)
    REPORT['started_utc'] = datetime.now(timezone.utc).isoformat()
    REPORT['budget'] = 'Existing memory-budget-before.txt: estimated 3-7 GiB, not a bound. Add frozen ref LoRA (~4.16 MiB FP32).'
    save_report()  # Written before model loading/training.
    (OUT / 'nvidia-smi-before.txt').write_bytes(subprocess.check_output(['nvidia-smi']))
    monitor_file = (OUT / 'nvidia-smi-samples.csv').open('w')
    monitor = subprocess.Popen([
        'nvidia-smi', '--query-gpu=timestamp,name,memory.total,memory.used,utilization.gpu',
        '--format=csv', '-lms', '500'
    ], stdout=monitor_file, stderr=subprocess.STDOUT)

    revision = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
    print('Downloading pinned model snapshot (approximately 3 GB)...', flush=True)
    snapshot = snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', revision=revision,
        allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
    # Soup 0.75.1 deliberately skips symlinked safetensors. HF cache snapshots
    # contain links to blobs: materialize regular files without changing Soup.
    source = Path(snapshot)
    local = args.model_cache.resolve() / ('qwen-1.5b-' + revision)
    local.mkdir(parents=True, exist_ok=True)
    def sha256_file(path):
        with path.open('rb') as handle:
            return hashlib.file_digest(handle, 'sha256').hexdigest()
    materialized = {}
    for src in sorted(source.iterdir()):
        if not src.is_file():
            continue
        dst = local / src.name
        assert not dst.is_symlink(), f'Unexpected destination symlink: {dst}'
        src_hash = sha256_file(src)
        if not dst.is_file() or sha256_file(dst) != src_hash:
            shutil.copyfile(src, dst, follow_symlinks=True)
        assert sha256_file(dst) == src_hash, f'Copy mismatch: {src.name}'
        materialized[src.name] = {'sha256': src_hash, 'bytes': dst.stat().st_size}
    assert any(local.glob('*.safetensors')), 'Missing materialized weights'
    (OUT / 'materialized-files.json').write_text(json.dumps(materialized, indent=2))
    snapshot = str(local)
    print('Verified regular-file snapshot:', snapshot, flush=True)
    REPORT['model_revision'] = revision
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    for split in ('train', 'test'):
        original = read_rows(DATA / f'{split}.jsonl')
        converted = []
        for row in original:
            assert row['chosen'][:-1] == row['rejected'][:-1]
            converted.append({'prompt': row['chosen'][:-1],
                              'chosen': row['chosen'][-1:], 'rejected': row['rejected'][-1:]})
        path = OUT / f'{split}-conversational.jsonl'
        path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in converted), encoding='utf-8')
        REPORT[f'{split}_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()

    cfg_dict = yaml.safe_load(DRAFT.read_text())
    cfg_dict['base'] = snapshot
    cfg_dict['data']['train'] = str(OUT / 'train-conversational.jsonl')
    cfg_dict['output'] = str(OUT / 'adapter')
    cfg_path = OUT / 'soup.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg_dict, sort_keys=False))
    cfg = load_config(cfg_path)
    dataset = load_dataset(cfg.data)
    assert len(dataset['train']) == 500 and not dataset.get('val')
    assert isinstance(dataset['train'][0]['prompt'], list)
    torch.cuda.reset_peak_memory_stats()
    wrapper = DPOTrainerWrapper(cfg, report_to='none')
    wrapper.setup(dataset)
    trainer, model = wrapper.trainer, wrapper.model
    assert wrapper._stream_runtime is not None, 'Streaming did not activate'
    REPORT['stream_runtime'] = wrapper._stream_runtime.stats()

    # Compare ALL trainer token IDs after Soup's cap with independent template rendering.
    for raw, actual in zip(dataset['train'], trainer.train_dataset, strict=True):
        p = tokenizer.apply_chat_template(raw['prompt'], tokenize=True, return_dict=False, add_generation_prompt=True)
        assert actual['prompt_ids'] == p and len(p) <= 256
        for side in ('chosen', 'rejected'):
            full = tokenizer.apply_chat_template(raw['prompt'] + raw[side], tokenize=True, return_dict=False)
            assert full[:len(p)] == p
            assert p + actual[f'{side}_ids'] == full, f'{side}: unexpected template or truncation'
            assert len(full) <= 512 and tokenizer.eos_token_id in actual[f'{side}_ids']
    REPORT['trainer_token_audit'] = 'PASS: 500 pairs, exact template IDs, EOS and no truncation'
    align_trainable_dtype_for_fp16(model, fp16=trainer.args.fp16, bf16=trainer.args.bf16)
    assert trainer.args.fp16 and not trainer.args.bf16
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert sum(p.numel() for p in trainable.values()) == 1089536
    assert all('lora_' in n and '.default.' in n and not p.is_meta for n,p in trainable.items())
    REPORT['trainable_params'] = sum(p.numel() for p in trainable.values())
    (OUT / 'trainable.json').write_text(json.dumps({n: {'shape': list(p.shape), 'dtype': str(p.dtype)} for n,p in trainable.items()}, indent=2))
    before = {n: p.detach().float().cpu().clone() for n,p in trainable.items()}
    reference = {n: p.detach().cpu().clone() for n,p in model.named_parameters() if '.ref.' in n}
    assert reference and all(not p.requires_grad for n,p in model.named_parameters() if '.ref.' in n)
    save_file(before, str(OUT / 'adapter-before.safetensors'))

    probe_ids = [trainer.train_dataset[i]['prompt_ids'] for i in range(3)]
    def probe(disabled=False):
        model.eval()
        outputs = []
        with torch.no_grad(), (model.disable_adapter() if disabled else contextlib.nullcontext()):
            for ids in probe_ids:
                x = torch.tensor([ids], device='cuda')
                y = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False,
                          logits_to_keep=1).logits[:, -1].float().cpu()
                assert torch.isfinite(y).all()
                outputs.append(y)
        torch.cuda.synchronize()
        return torch.cat(outputs)

    baseline = probe()
    noise = (baseline - probe()).abs().max().item()
    base_disabled = probe(True)
    REPORT['repeat_logit_noise'] = noise
    REPORT['initial_adapter_effect'] = (baseline-base_disabled).abs().max().item()
    assert REPORT['initial_adapter_effect'] <= max(1e-4, 2*noise)
    REPORT['setup_and_probe_peak_allocated'] = torch.cuda.max_memory_allocated()
    if not FULL_RUN:
        trainer.args.max_steps = 8  # Bounded AMP diagnostic only; full run keeps epoch-based length.
    trainer.args.logging_nan_inf_filter = False
    trainer.args.save_strategy = 'no'
    trainer.args.disable_tqdm = True
    (OUT / 'trainer-args.json').write_text(trainer.args.to_json_string())
    trainer.create_optimizer()
    opt_ids = {id(p) for g in trainer.optimizer.param_groups for p in g['params']}
    assert opt_ids == {id(p) for p in trainable.values()}
    grad_events = []
    class GradAudit(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            grads = [p.grad for p in trainable.values() if p.grad is not None]
            finite = bool(grads) and all(torch.isfinite(g).all().item() for g in grads)
            nonzero = finite and any(torch.count_nonzero(g).item() for g in grads)
            scaler = trainer.accelerator.scaler
            event = {'step': state.global_step, 'finite': finite, 'nonzero': bool(nonzero),
                     'grad_tensor_count': len(grads),
                     'missing_gradients': [n for n,p in trainable.items() if p.grad is None],
                     'nonfinite_gradients': [n for n,p in trainable.items()
                         if p.grad is not None and not torch.isfinite(p.grad).all().item()],
                     'scale_before': scaler.get_scale() if scaler is not None else None,
                     'learning_rates': [g['lr'] for g in trainer.optimizer.param_groups]}
            self.pre_step = {n: p.detach().float().cpu().clone() for n,p in trainable.items()}
            grad_events.append(event)
            (OUT / 'gradients.json').write_text(json.dumps(grad_events, indent=2))
            assert grads, 'All gradients are missing'
            if finite:
                assert nonzero, 'All finite gradients are zero'
            else:
                assert scaler is not None and scaler.is_enabled(), 'Nonfinite gradients without active GradScaler'
            # Let GradScaler handle overflow, then independently verify a skipped
            # step made NO parameter change. Nonfinite applied updates still fail.
        def on_step_end(self, args, state, control, **kwargs):
            event = grad_events[-1]
            scaler = trainer.accelerator.scaler
            event['skipped'] = bool(trainer.accelerator.optimizer_step_was_skipped)
            event['scale_after'] = scaler.get_scale() if scaler is not None else None
            current = {n: p.detach().float().cpu() for n,p in trainable.items()}
            event['parameters_finite'] = all(torch.isfinite(p).all().item() for p in current.values())
            event['max_update'] = max((current[n]-self.pre_step[n]).abs().max().item() for n in current)
            event['actual_update'] = event['parameters_finite'] and event['max_update'] > 0 and not event['skipped']
            REPORT['gradient_events'] = grad_events
            REPORT['actual_updates'] = sum(e.get('actual_update', False) for e in grad_events)
            (OUT / 'gradients.json').write_text(json.dumps(grad_events, indent=2))
            save_report()
            print('AMP_STEP_AUDIT:', json.dumps(event), flush=True)
            assert event['parameters_finite'], 'Optimizer corrupted adapter parameters'
            if not event['finite']:
                assert event['skipped'] and event['max_update'] == 0, 'Nonfinite gradient step was applied'
                assert event['scale_after'] < event['scale_before'], 'Scaler did not back off'
            if event['skipped']:
                assert event['max_update'] == 0, 'Skipped step changed parameters'
            elif any(lr > 0 for lr in event['learning_rates']):
                assert event['actual_update'], 'Non-skipped positive-LR step did not change weights'
            if not FULL_RUN and REPORT['actual_updates'] >= 2:
                control.should_training_stop = True
            return control
    trainer.add_callback(GradAudit())
    # Independent presented-token counter. No worker processes or evaluation
    # loaders: these are training batches, including attempts skipped by AMP.
    assert trainer.args.dataloader_num_workers == 0
    counters = {'pairs_presented': 0, 'nonpadding_tokens_presented': 0, 'completion_tokens_presented': 0}
    original_collator = trainer.data_collator
    def counting_collator(examples):
        batch = original_collator(examples)
        counters['pairs_presented'] += len(examples)
        counters['nonpadding_tokens_presented'] += int(batch['attention_mask'].sum().item())
        counters['completion_tokens_presented'] += int(batch['completion_mask'].sum().item())
        return batch
    trainer.data_collator = counting_collator
    torch.cuda.reset_peak_memory_stats()
    model.train()
    print('FULL_DPO_V1: ONE FULL EPOCH, fresh adapter, 500 pairs, accumulation=8.' if FULL_RUN else
          'AMP_SKIP_AUDIT_V2: at most 8 attempts; stop after 2 actual updates. Warmup unchanged.', flush=True)
    torch.cuda.synchronize()
    start_training = time.perf_counter()
    result = trainer.train()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_training
    REPORT['independent_throughput'] = dict(counters, wall_seconds=elapsed,
        pairs_per_second=counters['pairs_presented']/elapsed,
        nonpadding_tokens_per_second=counters['nonpadding_tokens_presented']/elapsed,
        definition='Collated pair presentations, including AMP-skipped attempts; chosen/rejected prompts each counted; reference compute not counted as extra tokens')
    REPORT['training_peak_allocated'] = torch.cuda.max_memory_allocated()
    REPORT['training_peak_reserved'] = torch.cuda.max_memory_reserved()
    REPORT['train_metrics'] = result.metrics
    REPORT['steps'] = trainer.state.global_step
    (OUT / 'trainer-history.json').write_text(json.dumps(trainer.state.log_history, indent=2))
    assert REPORT.get('actual_updates', 0) >= 2, 'Fewer than two actual updates'
    if FULL_RUN:
        assert counters['pairs_presented'] == 500, 'Full run did not present exactly 500 pairs'
        assert trainer.state.global_step == 63, 'Expected ceil(500/8)=63 attempted steps'
    deltas = {n: (p.detach().float().cpu()-before[n]).abs().max().item() for n,p in trainable.items()}
    REPORT['max_parameter_delta'] = max(deltas.values())
    assert all(__import__('math').isfinite(v) for v in deltas.values()) and max(deltas.values()) > 0
    params = dict(model.named_parameters())
    assert all(torch.equal(params[n].detach().cpu(), value) for n,value in reference.items()), 'Reference adapter changed'
    after = probe()
    disabled_after = probe(True)
    REPORT['policy_logit_delta'] = (after-baseline).abs().max().item()
    REPORT['adapter_effect_after'] = (after-disabled_after).abs().max().item()
    REPORT['disabled_base_drift'] = (disabled_after-base_disabled).abs().max().item()
    threshold = max(1e-4, 10*noise)
    assert REPORT['policy_logit_delta'] > threshold, 'Update below probe noise floor'
    assert REPORT['adapter_effect_after'] > threshold, 'Adapter has no measured effect'
    assert REPORT['disabled_base_drift'] <= max(1e-4, 2*noise), 'Disabled base drifted'
    trainer.save_model(str(OUT / 'adapter'))
    tokenizer.save_pretrained(OUT / 'adapter')
    save_file({'before': baseline, 'after': after, 'disabled_after': disabled_after}, str(OUT / 'probe-logits.safetensors'))
    prefix = 'FULL_RUN_PASS' if FULL_RUN else 'SMOKE_PASS'
    REPORT['status'] = (prefix + '_WITH_SKIPPED_STEPS' if any(e['skipped'] for e in grad_events) else prefix)
    REPORT['limitations'] = 'No held-out quality claim; no reload parity yet; no resident gradient parity; no ship verdict.'

if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        REPORT['status'] = 'FAILED'
        REPORT['error'] = repr(exc)
        (OUT / 'traceback.txt').write_text(traceback.format_exc())
        raise
    finally:
        if 'torch' in sys.modules:
            torch = sys.modules['torch']
            if torch.cuda.is_initialized():
                REPORT['final_peak_allocated'] = torch.cuda.max_memory_allocated()
                REPORT['final_peak_reserved'] = torch.cuda.max_memory_reserved()
        if monitor is not None:
            monitor.terminate()
            monitor.wait(timeout=15)
        if monitor_file is not None:
            monitor_file.close()
        REPORT['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save_report()
        print(json.dumps(REPORT, indent=2, default=str), flush=True)
