
import json
import hashlib
import math
import sys
import unicodedata
from pathlib import Path
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer

out = Path(sys.argv[1])
model = "Qwen/Qwen2.5-1.5B-Instruct"
model_rev = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
dataset = "trl-lib/ultrafeedback_binarized"
data_rev = "47124cb5778f5d50de1c7676a412828f3ea7c555"

tok = AutoTokenizer.from_pretrained(model, revision=model_rev)
seen_prompts = set()
report = {
    "model": model, "model_revision": model_rev,
    "dataset": dataset, "dataset_revision": data_rev,
    "seed": 42, "max_full_tokens": 512, "max_prompt_tokens": 256,
    "selection": "streaming shuffle, buffer_size=2000; short single-turn pairs",
    "splits": {}
}

def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())

def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

for split, target in (("train", 500), ("test", 100)):
    source = load_dataset(
        dataset, revision=data_rev, split=split, streaming=True
    ).shuffle(seed=42, buffer_size=2000)

    rows, provenance, lengths = [], [], []
    counts = Counter()

    for index, row in enumerate(source):
        if index >= 20000:
            break
        counts["scanned"] += 1
        chosen, rejected = row["chosen"], row["rejected"]

        # Ограничиваемся диалогами user -> assistant с общим запросом.
        if (
            len(chosen) != 2 or len(rejected) != 2
            or [m["role"] for m in chosen] != ["user", "assistant"]
            or [m["role"] for m in rejected] != ["user", "assistant"]
            or chosen[0] != rejected[0]
        ):
            counts["unsupported_structure"] += 1
            continue

        prompt = chosen[0]["content"]
        good, bad = chosen[1]["content"], rejected[1]["content"]
        if not all(isinstance(x, str) and x.strip()
                   for x in (prompt, good, bad)):
            counts["empty_or_invalid"] += 1
            continue
        if normalized(good) == normalized(bad):
            counts["identical_answers"] += 1
            continue

        sc, sr = float(row["score_chosen"]), float(row["score_rejected"])
        if not (math.isfinite(sc) and math.isfinite(sr) and sc > sr):
            counts["non_strict_or_invalid_preference"] += 1
            continue

        prompt_key = digest(normalized(prompt))
        if prompt_key in seen_prompts:
            counts["duplicate_prompt"] += 1
            continue

        pids = tok.apply_chat_template(
            chosen[:1], tokenize=True, add_generation_prompt=True
        )
        cids = tok.apply_chat_template(
            chosen, tokenize=True, add_generation_prompt=False
        )
        rids = tok.apply_chat_template(
            rejected, tokenize=True, add_generation_prompt=False
        )
        if len(pids) > 256 or max(len(cids), len(rids)) > 512:
            counts["too_long"] += 1
            continue
        if cids[:len(pids)] != pids or rids[:len(pids)] != pids:
            counts["template_prefix_mismatch"] += 1
            continue
        if (
            tok.eos_token_id not in cids[len(pids):]
            or tok.eos_token_id not in rids[len(pids):]
        ):
            counts["missing_completion_eos"] += 1
            continue

        seen_prompts.add(prompt_key)
        # Сохраняем исходный разговорный формат пар.
        rows.append({"chosen": chosen, "rejected": rejected})
        provenance.append({
            "shuffled_index": index, "prompt_sha256": prompt_key,
            "score_chosen": sc, "score_rejected": sr
        })
        lengths.append({
            "prompt": len(pids),
            "chosen": len(cids) - len(pids),
            "rejected": len(rids) - len(pids),
            "full_max": max(len(cids), len(rids))
        })
        if len(rows) == target:
            break

    path = out / f"{split}.jsonl"
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8"
    )
    (out / f"{split}_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    stats = {
        "accepted": len(rows), "counts": dict(counts),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "mean_tokens": {
            key: round(sum(x[key] for x in lengths) / len(lengths), 2)
            for key in ("prompt", "chosen", "rejected")
        } if lengths else {},
        "max_full_tokens": max((x["full_max"] for x in lengths), default=0),
        "chosen_longer_fraction": (
            sum(x["chosen"] > x["rejected"] for x in lengths) / len(lengths)
        ) if lengths else None,
    }
    report["splits"][split] = stats
    (out / "data_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(split, json.dumps(stats, indent=2), flush=True)
    assert len(rows) == target, (
        f"Only {len(rows)}/{target} pairs collected. Inspect filters first."
    )

print("DONE: train=500, test=100; normalized prompt overlap=0")
print("Files:", out)
