"""Timestamp stdout/stderr and preserve the child's exit status (no shell)."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--log', required=True, type=Path)
parser.add_argument('command', nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ['--'] else args.command
if not command:
    parser.error('provide command after --')
args.log.parent.mkdir(parents=True, exist_ok=True)
# Exclusive creation prevents accidental loss of a previous failure log.
with args.log.open('x', encoding='utf-8') as log:
    def emit(text):
        log.write(datetime.now(timezone.utc).isoformat()+' '+text+'\n')
        log.flush()
    emit('COMMAND '+repr(command))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='', flush=True)
        emit(line.rstrip('\n'))
    code = process.wait()
    emit('EXIT '+str(code))
sys.exit(code)
