
import json
import sys
from pathlib import Path
from huggingface_hub import HfApi
from transformers import AutoConfig, AutoTokenizer
from datasets import load_dataset

out = Path(sys.argv[1])
model_id = "Qwen/Qwen2.5-1.5B-Instruct"
dataset_id = "trl-lib/ultrafeedback_binarized"

api = HfApi()
model_revision = api.model_info(model_id).sha
dataset_revision = api.dataset_info(dataset_id).sha

manifest = {
    "model": model_id,
    "model_revision": model_revision,
    "dataset": dataset_id,
    "dataset_revision": dataset_revision,
}
(out / "sources.json").write_text(json.dumps(manifest, indent=2))
print("SOURCES:", json.dumps(manifest, indent=2))

config = AutoConfig.from_pretrained(model_id, revision=model_revision)
tokenizer = AutoTokenizer.from_pretrained(
    model_id, revision=model_revision
)
config.save_pretrained(out / "model_metadata")
tokenizer.save_pretrained(out / "tokenizer")

for key in (
    "model_type", "hidden_size", "intermediate_size",
    "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "vocab_size", "tie_word_embeddings"
):
    print(f"{key}: {getattr(config, key, None)}")

print("EOS:", repr(tokenizer.eos_token), tokenizer.eos_token_id)
print("PAD:", repr(tokenizer.pad_token), tokenizer.pad_token_id)
print("Chat template present:", bool(tokenizer.chat_template))

data = load_dataset(
    dataset_id, revision=dataset_revision, streaming=True
)
print("SPLITS:", list(data))

for split, rows in data.items():
    example = next(iter(rows))
    text = json.dumps(example, ensure_ascii=False, indent=2)
    (out / f"example_{split}.json").write_text(text, encoding="utf-8")
    print(f"\nSPLIT: {split}")
    print("COLUMNS:", list(example))
    print("EXAMPLE (preview):", text[:2500])
