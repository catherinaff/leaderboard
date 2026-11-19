from pathlib import Path
import subprocess
import sys

p=Path(__file__).resolve().parent
configs={
 'runtime_onellm_20261113':['fairscale==0.4.13','open_clip_torch==2.24.0','timm==0.9.16','torchvision==0.24.0','sentencepiece==0.2.0'],
 'runtime_sensorllm_20261113':['transformers==4.45.2','tokenizers==0.20.3','peft==0.13.2','accelerate==1.0.1','sentencepiece==0.2.0','tensorboard==2.18.0','neurokit2==0.2.10','tiktoken==0.8.0'],
}
for name,packages in configs.items():
 subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(p/name)],check=True)
 subprocess.run([str(p/name/'bin/python'),'-m','pip','install','--no-deps',*packages],check=True)
 print('COMPLETE',name,flush=True)
