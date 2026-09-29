"""CPU-only audit of packaged evidence. Does not rerun training or model inference."""
from pathlib import Path
import csv
import hashlib
import json
import math

root=Path(__file__).resolve().parent
for entry in json.loads((root/'runs/manifest.json').read_text()):
    p=root/entry['file']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256'], p
full=root/'runs/20260929T202553230689Z-full-dpo'
ev=root/'runs/20260929T210735564189Z-heldout-eval'
train=json.loads((full/'smoke-report.json').read_text())
evaluation=json.loads((ev/'evaluation-report.json').read_text())
events=train['gradient_events']
assert len(events)==63 and sum(e['actual_update'] for e in events)==60
assert sum(e['skipped'] for e in events)==2
assert all(e['max_update']==0 for e in events if e['skipped'])
assert train['independent_throughput']['pairs_presented']==500
scores=[json.loads(s) for s in (ev/'heldout-rows.jsonl').read_text().splitlines()]
assert len(scores)==100
assert sum(r['base_correct'] for r in scores)==sum(r['tuned_correct'] for r in scores)==56
assert all(r['base_correct']==r['tuned_correct'] for r in scores)
assert json.loads((ev/'ship-verdict.json').read_text())['decision']=="DON'T SHIP"
rows=list(csv.DictReader((full/'nvidia-smi-samples.csv').open()))
peak=max(int(r[' memory.used [MiB]'].split()[0]) for r in rows)
assert peak==7189
reload=json.loads((root/'runs/20260929T210735599570Z-reload-check/reload-report.json').read_text())
assert reload['exact_saved_tensor_matches']==112
assert reload['stream_vs_resident_active_max_error']==reload['stream_vs_resident_disabled_max_error']==0
print("EVIDENCE AUDIT PASS: checksums; 500 pairs; 60 updates; 2 AMP skips; 56/100 unchanged; reload exact; nvidia peak 7189 MiB; DON'T SHIP.")
