"""Compile the author's unchanged simulator in a disposable work directory."""
from pathlib import Path
import json
import shutil
import subprocess
import sys

p=Path(__file__).resolve().parent
binary=p/'assets_20261113/imugpt/source_data/imusim'
if not binary.exists():raise FileNotFoundError('Exact source archive C/pyx assets are not yet ready')
work=p/'build_workdirs_20261113/imusim'
if work.exists():shutil.rmtree(work)
shutil.copytree(p/'vendor/imugpt/imusim',work)
shutil.copytree(binary,work,dirs_exist_ok=True)
shutil.copyfile(work/'setup_new.py',work/'setup.py')
subprocess.run([sys.executable,'-m','pip','install','SimPy==2.3.1','pyparsing','Cython==0.29.37'],check=True)
subprocess.run([sys.executable,'-m','pip','install','--no-deps','--no-build-isolation',str(work)],check=True)
subprocess.run([sys.executable,'-c','from imusim.maths.quaternions import Quaternion;from imusim.platforms.imus import IdealIMU;print("IMUSim imports verified")'],check=True)
(p/'audits/imugpt_simulator_runtime_20261113.json').write_text(json.dumps(dict(status='ready',source='Immutable author source; build workdir copy; source in vendor unchanged',interpreter=sys.executable),indent=2)+'\n')
