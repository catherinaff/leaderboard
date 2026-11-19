from pathlib import Path
import os,json,hashlib,shutil
os.environ["HF_HUB_DISABLE_XET"]="1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"]="120"
from huggingface_hub import HfApi,hf_hub_download
p=Path("/root/pengrf/leadiboard/iotbench_har/experiments/imu_continuation_20261106/assets_20261113/onellm")
p.mkdir(parents=True,exist_ok=True)
assert shutil.disk_usage(p).free>38*2**30,"Keep at least 32 GiB free during shared model staging"
api=HfApi(endpoint="https://hf-mirror.com");info=api.model_info("csuhan/OneLLM-7B",files_metadata=True,timeout=15)
manifest=dict(repo="csuhan/OneLLM-7B",revision=info.sha,files={})
for name in ["params.json","tokenizer.model","consolidated.00-of-01.pth"]:
 file=Path(hf_hub_download("csuhan/OneLLM-7B",name,revision=info.sha,local_dir=p,endpoint="https://hf-mirror.com"))
 h=hashlib.sha256()
 with file.open("rb") as stream:
  for chunk in iter(lambda:stream.read(8*1024*1024),b""):h.update(chunk)
 original=next(f for f in info.siblings if f.rfilename==name)
 if original.lfs:assert h.hexdigest()==original.lfs.sha256
 manifest["files"][name]=dict(sha256=h.hexdigest(),size=file.stat().st_size)
 print("OneLLM asset",name,file.stat().st_size,flush=True)
(p/"ASSETS.json").write_text(json.dumps(manifest,indent=2)+"\n")
print("OneLLM author checkpoint ready",flush=True)
