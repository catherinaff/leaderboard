from pathlib import Path
import subprocess,os
p=(Path(__file__).resolve().parent)
for dataset,fold in [("uci_har","official"),("usc_had","test_1_val_2"),("pamap2","test_101_val_102"),("realworld_har","test_1_val_2")]:
 with (p/"logs"/("imu2clip__"+dataset+"__adapter_smoke_20261113.log")).open("ab") as log:
  args=[str(p/"runtime_llm_20261113/bin/python"),"-u",str(p/"train.py"),"--method","imu2clip","--dataset",dataset,"--seed","42","--fold",fold,"--gpu","0","--memory-fraction",".16","--batch-size","4","--smoke","--epochs","1"]
  result=subprocess.run(args,cwd=p,env=dict(os.environ,IOTBENCH_MATH_SDPA="1",OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4"),stdout=log,stderr=subprocess.STDOUT)
 print(dataset,"exit",result.returncode,flush=True)
