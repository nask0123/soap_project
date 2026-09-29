"""Independent resident reload check: RELOAD_AUTOCAST_AUDIT_V2."""
import contextlib
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/content/soup_takehome')
SOURCE = ROOT / '20260929T201018626035Z-smoke'
OUT = ROOT / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-reload-check')
OUT.mkdir(parents=True)
report = {'status': 'STARTED', 'source': str(SOURCE), 'output': str(OUT)}

def main():
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    prior = json.loads((SOURCE / 'smoke-report.json').read_text())
    assert prior['status'].startswith('SMOKE_PASS')
    adapter = SOURCE / 'adapter'
    acfg = json.loads((adapter / 'adapter_config.json').read_text())
    base_path = acfg['base_model_name_or_path']
    assert '989aa7980e4cf806f80c7fef2b1adb7bc71aa306' in base_path
    tok = AutoTokenizer.from_pretrained(adapter)
    model = AutoModelForCausalLM.from_pretrained(base_path, dtype=torch.float16, device_map='cuda')
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model.eval()
    assert not any(p.requires_grad for p in model.parameters())
    # Check every saved adapter tensor was loaded with its exact value.
    saved = load_file(str(adapter / 'adapter_model.safetensors'))
    loaded = {n.replace('.default.', '.'): p for n,p in model.named_parameters() if 'lora_' in n}
    assert set(saved) == set(loaded), {'missing': sorted(set(saved)-set(loaded)), 'extra': sorted(set(loaded)-set(saved))}
    assert all(torch.equal(saved[n], loaded[n].detach().cpu()) for n in saved), 'Saved weights not loaded exactly'
    report['exact_saved_tensor_matches'] = len(saved)

    rows = [json.loads(s) for s in (SOURCE / 'train-conversational.jsonl').read_text().splitlines()][:3]
    probes = [tok.apply_chat_template(r['prompt'], tokenize=True, return_dict=False, add_generation_prompt=True) for r in rows]
    expected = load_file(str(SOURCE / 'probe-logits.safetensors'))
    def run(disabled=False, amp=False):
        outputs = []
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=amp), (model.disable_adapter() if disabled else contextlib.nullcontext()):
            for ids in probes:
                x = torch.tensor([ids], device='cuda')
                outputs.append(model(input_ids=x, attention_mask=torch.ones_like(x),
                    use_cache=False, logits_to_keep=1).logits[:, -1].float().cpu())
        torch.cuda.synchronize()
        result = torch.cat(outputs)
        assert torch.isfinite(result).all()
        return result

    # Preserve the original no-autocast result as a control, then match the
    # FP16 autocast used by Accelerator's training-prepared forward.
    no_amp_active = run()
    no_amp_disabled = run(True)
    report['no_autocast_control'] = {
        'active_max_error': (no_amp_active-expected['after']).abs().max().item(),
        'disabled_max_error': (no_amp_disabled-expected['disabled_after']).abs().max().item(),
        'active_allclose': torch.allclose(no_amp_active, expected['after'], atol=0.001, rtol=0.001),
    }
    active = run(amp=True)
    repeat = run(amp=True)
    disabled = run(True, amp=True)
    disabled_repeat = run(True, amp=True)
    report['comparison_mode'] = 'explicit CUDA FP16 autocast; unchanged atol/rtol'
    report['adapter_parameter_dtypes'] = sorted({str(p.dtype) for p in loaded.values()})
    from safetensors.torch import save_file
    save_file({'no_amp_active': no_amp_active, 'amp_active': active,
               'amp_disabled': disabled}, str(OUT / 'reload-logits.safetensors'))
    noise = (active-repeat).abs().max().item()
    effect = (active-disabled).abs().max().item()
    report.update({
        'repeat_noise_max': noise,
        'reloaded_adapter_effect_max': effect,
        'stream_vs_resident_active_max_error': (active-expected['after']).abs().max().item(),
        'stream_vs_resident_disabled_max_error': (disabled-expected['disabled_after']).abs().max().item(),
        'parity_tolerance': {'atol': 0.001, 'rtol': 0.001},
        'active_allclose': torch.allclose(active, expected['after'], atol=0.001, rtol=0.001),
        'disabled_allclose': torch.allclose(disabled, expected['disabled_after'], atol=0.001, rtol=0.001),
        'peak_allocated': torch.cuda.max_memory_allocated(),
        'peak_reserved': torch.cuda.max_memory_reserved(),
    })
    threshold = max(0.0001, 10*noise)
    report['effect_threshold'] = threshold
    report['negative_control_detected'] = not ((disabled-disabled_repeat).abs().max().item() > threshold)
    assert effect > threshold, 'Loaded adapter has no measurable effect above repeat noise'
    assert report['active_allclose'] and report['disabled_allclose'], 'Resident reload differs from streamed output'
    assert report['negative_control_detected']
    report['status'] = 'RELOAD_PASS_MATCHED_AUTOCAST'
    report['limitations'] = 'Three training prompts, last-token logits only; no held-out quality, gradient parity or deployment validation. Allclose is tolerance-based, not bitwise equality.'

if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        report['status'] = 'FAILED'
        report['error'] = repr(exc)
        (OUT / 'traceback.txt').write_text(traceback.format_exc())
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        (OUT / 'reload-report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
