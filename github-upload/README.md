# Soup DPO take-home

**Verdict: DON'T SHIP.** The adapter changed and survives saving/reloading, but held-out preference accuracy did not improve. Russian support quality and forgetting remain unmeasured.

[Two-page report](output/pdf/report.pdf) · [Report source](REPORT.md) · [Audit notes and failures](AUDIT_NOTES.md)

[Open reproduction notebook in Colab](https://colab.research.google.com/github/nask0123/soap_project/blob/main/reproduce.ipynb)

## Measured results

| Check | Result |
|---|---|
| Hardware | Tesla T4, 15,360 MiB |
| Training | 500 preference pairs, one epoch, layer streaming |
| Optimizer | 63 attempts: 60 updates, 1 zero-LR warmup, 2 AMP overflow skips |
| Peak GPU memory | 3.33 GiB allocated; 6.86 GiB reserved; sampled nvidia-smi 7,189 MiB |
| Held-out ranking | 56/100 before and after; no preference flips |
| Reload check | 112 saved tensors match; streamed/resident probe error 0 with matched FP16 autocast |
| soup ship | Exit 2, DON'T SHIP: missing independent baseline |

The reload check covers three training prompts and last-token logits. It does not establish gradient parity, held-out quality, or deployment correctness. A working adapter alone is not a shipping criterion.

## Deliverables and raw evidence

- `runs/`: original timestamped logs, failed attempts, raw nvidia-smi, configs, adapters and evaluation outputs. [Manifest](runs/manifest.json) records SHA-256 hashes of 156 original files.
- [Full run](runs/20260929T202553230689Z-full-dpo), [held-out evaluation](runs/20260929T210735564189Z-heldout-eval), [reload verification](runs/20260929T210735599570Z-reload-check).
- `smoke_dpo.py`: instrumented Soup DPO training, including optimizer/gradient checks.
- `verify_adapter.py`: independent saved-adapter verification and disabled-adapter negative control.
- `evaluate_run.py`: held-out scores, length sensitivity, paired bootstrap and soup ship.
- `data/`, `prepare_data.py`, `config/soup.yaml`: prepared data, provenance, preparation and configuration.
- `requirements-colab.lock.txt`: exact installed package versions captured in Colab.

Original measured scripts remain inside timestamped launch directories. Root scripts add portable paths and command-line arguments. They received local syntax/CLI checks; those portability edits and the reproduction notebook have not been rerun on a GPU. All measured claims refer to the preserved original runs. The notebook is an unexecuted reproduction launcher, not a fabricated completed notebook; executed code and raw outputs are supplied separately.

## Reproduce

Use a Linux T4 runtime and Python 3.12. The Colab notebook creates an isolated interpreter and records subprocess logs. Base weights download separately (approximately 3 GB); allow extra disk for materialization and layer shards. The original run used about 13 GB host RAM availability; host peak RAM was not measured.

```bash
python -m pip install -r requirements-colab.lock.txt
python inspect_results.py
python run_logged.py --log runs/new-training.log -- python -u smoke_dpo.py --full
# Replace NEW-RUN with the newly printed full-dpo directory:
python run_logged.py --log runs/new-evaluation.log -- python -u evaluate_run.py runs/NEW-RUN
```

Evaluation runs reload verification in a separate process. `verify_adapter.py runs/NEW-RUN` can also be run independently. Log filenames must be new: the logger refuses to overwrite evidence. `inspect_results.py` verifies submitted evidence on CPU without training dependencies.

The runner converts the prepared string data into the conversational form required by the pinned tokenizer. Do not pass the template YAML directly to Soup as a substitute for the runner. Data regeneration is optional: create a new directory and run `python prepare_data.py PATH-TO-NEW-DIRECTORY`; keep the submitted data unchanged.

Model: [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Dataset: [UltraFeedback binarized](https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized), revision `47124cb5778f5d50de1c7676a412828f3ea7c555`. This is an open preference dataset, not Russian support-ticket evidence. [Soup upstream](https://github.com/MakazhanAlpamys/Soup).

## AI assistance and limitations

AI helped write scripts, investigate failures and draft the report. The candidate executed the Colab cells and supplied raw outputs. Accepted fixes included Python 3.12 isolation, checkpoint materialization and matched autocast. Incorrect tokenizer handling and a mismatched-precision reload test were corrected; failed evidence remains available. No GPU execution occurred on the local Windows host. See the report for what surprised us and remaining concerns. The candidate should review the interpretation before submission and attach their CV separately if requested.
