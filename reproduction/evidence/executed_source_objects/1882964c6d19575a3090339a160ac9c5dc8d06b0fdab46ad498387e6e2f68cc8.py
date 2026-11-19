from pathlib import Path
import os,json,subprocess,hashlib
from datetime import datetime,timezone
p=Path("/root/pengrf/leadiboard/iotbench_har/experiments/imu_continuation_20261106")
def state(status,**extra):
 f=p/"state/data_worker.json";tmp=f.with_suffix(".tmp");tmp.write_text(json.dumps(dict(status=status,pid=os.getpid(),updated_at=datetime.now(timezone.utc).isoformat(),**extra),indent=2)+"\n");tmp.replace(f)
try:
 state("verifying_frozen_preparation_source")
 subprocess.run(["/opt/conda/envs/fireredasr/bin/python","-u",str(p/"prepare_extra.py"),"--dataset","realworld_har"],check=True,cwd=p)
 before=json.loads((p/"archives/realworld_manifest_before_source_snapshot.json").read_text());after=json.loads((p/"data/realworld_har/manifest.json").read_text())
 assert before["arrays_sha256"]==after["arrays_sha256"],"Preparation source capture must not change any data array"
 assert hashlib.sha256((p/"data/realworld_har/preparation_source.py").read_bytes()).hexdigest()==after["preprocessing_sha256"]
 assert len(list((p/"manifests/splits/realworld_har").glob("seed_*/*.npz")))==75
 state("ready",dataset="realworld_har",source_snapshot_verified=True,repeated_arrays_identical=True)
except Exception as e:
 state("failed",error=str(e));raise
