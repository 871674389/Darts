# --- darts: make this script runnable from anywhere (repo-root bootstrap) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, 'model')):
    _ROOT = _os.path.dirname(_ROOT)
_sys.path.insert(0, _ROOT)
# ---------------------------------------------------------------------------
import torch,numpy as np,os
os.environ["DUALV4_DS"]="PSM"; os.environ["DUALV4_WINDOW"]="15"
from model.DualGraphSTNet_dualshortpath import DualGraphSTNet
from lib.utils import load_data_test
import config_files.config_general as cg
dev="cuda:0"
for F in [25,38,55,118]:
    import importlib; importlib.reload(cg); c=cg.Config()
    c.num_features=F; c.num_nodes=F; c.device=dev
    np.save("/tmp/dummy.npy",np.random.randn(2000,F).astype(np.float32))
    loader=load_data_test("/tmp/dummy.npy",c,dev)
    m=DualGraphSTNet(matrix_decay=c.matrix_decay,num_features=F,window_size=c.window_size,out_dim=c.out_dim,
        target_size=c.target_size,gru_dim=c.gru_dim,Graph_learner_n_hid=c.Graph_learner_n_hid,
        Graph_learner_n_head_dim=c.Graph_learner_n_head_dim,num_heads=c.num_heads,num_nodes=F,
        temperature=c.temperature,max_diffusion_step=c.max_diffusion_step,num_rnn_layers=c.num_rnn_layers,device=dev).to(dev)
    with torch.no_grad():
        iw,ps,_=next(iter(loader)); m(iw.to(dev),ps.to(dev))
    print(f"  {F}维: {sum(p.numel() for p in m.parameters()):,} 参数")
