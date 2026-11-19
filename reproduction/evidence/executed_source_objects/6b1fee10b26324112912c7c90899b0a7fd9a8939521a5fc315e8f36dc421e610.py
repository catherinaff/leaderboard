"""Incremental evidence audit for newly admitted methods, no score selection."""
from pathlib import Path
from datetime import datetime,timezone
import json

from audit_results_20261111 import verify,write

p=Path(__file__).resolve().parent
methods=['gpt4ts','units','calf','timellm','imu2clip','imugpt','sensorllm']
verified=[];errors=[]
for method in methods:
 for file in sorted((p/'runs'/method).glob('*/*/*/result.json')):
  try:verified.append(verify(file))
  except Exception as error:errors.append(dict(path=str(file),error=f'{type(error).__name__}: {error}'))
value=dict(status='passed' if not errors else 'failed',completed_at=datetime.now(timezone.utc).isoformat(),
           scope='Every newly complete supervised adapter run present at audit start; benchmark means still require every fold and seed.',verified=verified,errors=errors)
write(p/'audits/new_adapter_runs_20261113.json',value)
print(json.dumps(dict(status=value['status'],verified=len(verified),failed=len(errors),errors=errors),indent=2),flush=True)
