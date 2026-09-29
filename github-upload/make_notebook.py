"""Build an explicitly unexecuted Colab reproduction launcher."""
import json
from pathlib import Path

cells=[]
def cell(kind, text):
    item={'cell_type':kind,'metadata':{},'id':f'cell-{len(cells)}','source':text.splitlines(True)}
    if kind=='code': item.update(execution_count=None,outputs=[])
    cells.append(item)

cell('markdown', '''# Reproduce the Soup T4 experiment
Select **Runtime → Change runtime type → T4 GPU**. Run cells in order.
This notebook is an unexecuted launcher. Original executed scripts and raw results are in `runs/`.
The submitted verdict is DON'T SHIP. A successful rerun does not automatically change that verdict.
''')
cell('code', '''from pathlib import Path
import subprocess, sys
from datetime import datetime, timezone
REPO=Path('/content/soap_project')
if not REPO.exists():
    subprocess.run(['git','clone','https://github.com/nask0123/soap_project.git',str(REPO)],check=True)
subprocess.run(['nvidia-smi'],check=True)
subprocess.run([sys.executable,'-m','pip','install','uv'],check=True)
uv=[sys.executable,'-m','uv']
PY=str(REPO/'.venv/bin/python')
if not Path(PY).exists():
    subprocess.run([*uv,'venv','--python','3.12',str(REPO/'.venv')],check=True)
def logged(label, command):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    subprocess.run([sys.executable,str(REPO/'run_logged.py'),'--log',
                    str(REPO/'runs'/f'{stamp}-{label}.log'),'--',*command],cwd=REPO,check=True)
''')
cell('code', '''logged('install', [*uv,'pip','install','--python',PY,'-r',str(REPO/'requirements-colab.lock.txt')])
logged('evidence-audit',[PY,str(REPO/'inspect_results.py')])
''')
cell('markdown', 'The next cell starts a fresh one-epoch run, downloads the pinned model and saves new evidence. Earlier failures stay intact. Read the memory estimate in the report before training.')
cell('code', '''before=set((REPO/'runs').glob('*-full-dpo'))
logged('training',[PY,'-u',str(REPO/'smoke_dpo.py'),'--full'])
created=set((REPO/'runs').glob('*-full-dpo'))-before
assert len(created)==1, created
SOURCE=created.pop()
print('New run:',SOURCE)
''')
cell('code', '''logged('evaluation',[PY,'-u',str(REPO/'evaluate_run.py'),str(SOURCE)])
''')
cell('markdown', 'Read the new training, reload, evaluation and ship reports together. Evaluation may honestly return a negative shipping verdict. Preserve failures and do not retune on this test set.')
cell('code', '''import shutil
from google.colab import files
archive=shutil.make_archive('/content/soup-results','zip',REPO,'runs')
files.download(archive)
''')
notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'accelerator':'GPU'},'nbformat':4,'nbformat_minor':5}
Path(__file__).with_name('reproduce.ipynb').write_text(json.dumps(notebook,indent=2),encoding='utf-8')
