"""Random-weight TabICLv2 checkpoint with the released architecture, for CPU timing only (predictions meaningless).
The architecture is read from tabicl_v2_config.json, which the Colab notebook writes from the real checkpoint
(the released weights are downloaded there; only their configuration is needed here).
Usage: python make_timing_checkpoint_tabicl.py <tabicl_v2_config.json> [out.ckpt]"""
import sys, json, torch
from tabicl._model.tabicl import TabICL
cfg = json.load(open(sys.argv[1])); out = sys.argv[2] if len(sys.argv) > 2 else "timing_tabicl_v2_random_weights.ckpt"
torch.manual_seed(0); model = TabICL(**cfg)
torch.save({"config": cfg, "state_dict": model.state_dict()}, out)
print(f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters -> {out}")
