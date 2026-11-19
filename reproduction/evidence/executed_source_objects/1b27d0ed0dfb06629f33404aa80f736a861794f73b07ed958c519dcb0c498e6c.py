from pathlib import Path
import os,json,hashlib,shutil
os.environ["HF_HUB_DISABLE_XET"]="1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"]="120"
from huggingface_hub import HfApi,hf_hub_download
p=Path("/root/pengrf/leadiboard/iotbench_har/experiments/imu_continuation_20261106/assets_20261113/llama3")
p.mkdir(parents=True,exist_ok=True)
api=HfApi(endpoint="https://hf-mirror.com")
original=api.model_info("meta-llama/Meta-Llama-3-8B",files_metadata=True,timeout=15)
mirror=api.model_info("NousResearch/Meta-Llama-3-8B",files_metadata=True,timeout=15)
weights={f.rfilename:f.lfs.sha256 for f in original.siblings if f.rfilename.endswith("safetensors")}
other={f.rfilename:f.lfs.sha256 for f in mirror.siblings if f.rfilename.endswith("safetensors")}
assert weights==other and len(weights)==4,"Public redistribution must exactly match the official four weight hashes"
manifest=dict(repo=mirror.id,revision=mirror.sha,official_repo=original.id,official_revision=original.sha,official_weight_sha256=weights,files={})
(p/"VERIFIED_SOURCE.json").write_text(json.dumps(manifest,indent=2)+"\n")
assert shutil.disk_usage(p).free>52*2**30
files=[f.rfilename for f in mirror.siblings if f.rfilename in ["config.json","generation_config.json","model.safetensors.index.json","special_tokens_map.json","tokenizer.json","tokenizer_config.json"]]+list(weights)
for name in files:
 file=Path(hf_hub_download(mirror.id,name,revision=mirror.sha,local_dir=p,endpoint="https://hf-mirror.com"))
 h=hashlib.sha256()
 with file.open("rb") as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b""):h.update(block)
 if name in weights:assert h.hexdigest()==weights[name]
 manifest["files"][name]=dict(sha256=h.hexdigest(),size=file.stat().st_size)
 print("Llama3 verified asset",name,file.stat().st_size,flush=True)
(p/"ASSETS.json").write_text(json.dumps(manifest,indent=2)+"\n")
print("All original-equivalent Llama3 weights verified",flush=True)
