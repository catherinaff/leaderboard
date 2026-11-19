from pathlib import Path
import subprocess,json
p=(Path(__file__).resolve().parent)
for method in ["hart","mobilehart"]:
 with (p/"logs"/(method+"__usc_smoke.log")).open("ab") as log:
  args=["/opt/conda/envs/hart-tf/bin/python","-u",str(p/"train_tf.py"),"--method",method,"--dataset","usc_had","--seed","42","--fold","test_1_val_2","--gpu","0","--memory-fraction",".08","--smoke","--epochs","1","--batch-size","8"]
  result=subprocess.run(args,cwd=p,stdout=log,stderr=subprocess.STDOUT)
 print(method,"exit",result.returncode,flush=True)
