from pathlib import Path
import requests,hashlib,json,os,shutil
p=Path("/root/pengrf/leadiboard/iotbench_har/experiments/imu_continuation_20261106/assets_20261113/units")
p.mkdir(parents=True,exist_ok=True)
url="https://github.com/mims-harvard/UniTS/releases/download/ckpt/units_x64_supervised_checkpoint.pth"
file=p/"units_x64_supervised_checkpoint.pth"
if not file.exists():
 r=requests.get(url,stream=True,timeout=30);r.raise_for_status()
 temp=file.with_suffix(".incomplete")
 with temp.open("wb") as out:
  for chunk in r.iter_content(1024*1024):out.write(chunk)
 temp.replace(file)
sha=hashlib.sha256(file.read_bytes()).hexdigest()
(p/"ASSETS.json").write_text(json.dumps(dict(url=url,sha256=sha,size=file.stat().st_size),indent=2)+"\n")
print("UniTS pretrained asset ready",file.stat().st_size,sha,flush=True)
