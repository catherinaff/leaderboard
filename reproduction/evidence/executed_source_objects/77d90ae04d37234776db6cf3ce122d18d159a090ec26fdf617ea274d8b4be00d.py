from pathlib import Path
import numpy as np,torch,json,hashlib
from sklearn.decomposition import PCA
from transformers import GPT2Model
p=Path("/root/pengrf/leadiboard/iotbench_har/experiments/imu_continuation_20261106/assets_20261113/gpt2")
np.random.seed(42);torch.manual_seed(42)
model=GPT2Model.from_pretrained(str(p),local_files_only=True)
wte=model.wte.state_dict()["weight"].cpu().numpy()
basis=PCA(n_components=500).fit_transform(wte.T)
assert basis.shape==(768,500) and np.isfinite(basis).all()
file=p/"wte_pca_500.pt";torch.save(torch.from_numpy(basis),file)
(p/"PCA.json").write_text(json.dumps(dict(algorithm="Author pca.py: PCA(n_components=500).fit_transform(wte.T)",seed=42,shape=list(basis.shape),sha256=hashlib.sha256(file.read_bytes()).hexdigest(),serialization="Tensor values; preserves the original numeric basis while supporting modern torch weights_only loading"),indent=2)+"\n")
print("PCA500 author basis prepared",flush=True)
