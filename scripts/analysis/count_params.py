# --- darts: make this script runnable from anywhere (repo-root bootstrap) ---
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while _ROOT != _os.path.dirname(_ROOT) and not _os.path.isdir(_os.path.join(_ROOT, 'model')):
    _ROOT = _os.path.dirname(_ROOT)
_sys.path.insert(0, _ROOT)
# ---------------------------------------------------------------------------
import os,sys
sys.path.insert(0,".")
from model.DualGraphSTNet_dualshortpath import DualGraphSTNet
import importlib
for ds in ["MSL","SWAT","WADI","SMD","PSM"]:
    os.environ["DUALV4_DS"]=ds; os.environ["DUALV4_WINDOW"]="15"
    import config_files.config_general as cg; importlib.reload(cg)
    c=cg.Config()
    m=DualGraphSTNet(matrix_decay=c.matrix_decay,num_features=c.num_features,window_size=c.window_size,
        out_dim=c.out_dim,target_size=c.target_size,gru_dim=c.gru_dim,Graph_learner_n_hid=c.Graph_learner_n_hid,
        Graph_learner_n_head_dim=c.Graph_learner_n_head_dim,num_heads=c.num_heads,num_nodes=c.num_nodes,
        temperature=c.temperature,max_diffusion_step=c.max_diffusion_step,num_rnn_layers=c.num_rnn_layers,device="cpu")
    n=sum(p.numel() for p in m.parameters())
    print(f"  {ds}: {c.num_nodes}维 -> {n:,} 参数 ({n/1e3:.1f}K)")
