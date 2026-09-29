
import sys
import shutil
import importlib.util
from pathlib import Path
from importlib.metadata import version

out = Path(sys.argv[1])
for name in ("soup-cli", "transformers", "trl", "peft", "torch"):
    print(name, version(name))

# Копируем исходники, не импортируя модель и не загружая веса.
for package, folders in (
    ("soup_cli", ("config", "data", "trainer", "utils", "commands")),
    ("trl", ("trainer",)),
):
    spec = importlib.util.find_spec(package)
    assert spec and spec.submodule_search_locations, package
    base = Path(next(iter(spec.submodule_search_locations)))
    count = 0
    for folder in folders:
        for src in (base / folder).rglob("*.py"):
            dst = out / "installed_source" / package / src.relative_to(base)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            count += 1
    print(f"{package}: saved {count} Python source files")
