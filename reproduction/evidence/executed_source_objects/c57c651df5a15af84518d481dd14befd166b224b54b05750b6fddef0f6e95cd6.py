from pathlib import Path
import subprocess
import sys

p=Path(__file__).resolve().parent
subprocess.run([sys.executable,'-m','pip','install','openai==1.109.1'],check=True)
subprocess.run([sys.executable,'-u',str(p/'api_adapters.py'),'--dry-run-matrix'],check=True)
