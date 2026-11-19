from pathlib import Path
import os,json,hashlib,shutil
os.environ["HF_HUB_DISABLE_XET"]="1"
from huggingface_hub import hf_hub_download,HfApi
p=Path("/root/pengrf/leadiboard/iotbench_har/experiments/imu_continuation_20261106/assets_20261113/gpt2")
p.mkdir(parents=True,exist_ok=True)
assert shutil.disk_usage(p).free>66*2**30
files=["config.json","model.safetensors","merges.txt","vocab.json","tokenizer.json","tokenizer_config.json","generation_config.json"]
meta=HfApi(endpoint="https://hf-mirror.com").model_info("openai-community/gpt2",files_metadata=True,timeout=15)
manifest=dict(repo="openai-community/gpt2",revision=meta.sha,files={})
for name in files:
 file=Path(hf_hub_download("openai-community/gpt2",name,revision=meta.sha,local_dir=p,endpoint="https://hf-mirror.com"))
 digest=hashlib.sha256(file.read_bytes()).hexdigest()
 manifest["files"][name]=dict(size=file.stat().st_size,sha256=digest)
 print("asset",name,file.stat().st_size,flush=True)
(p/"ASSETS.json").write_text(json.dumps(manifest,indent=2)+"\n")
print("GPT2 assets ready",meta.sha,flush=True)
