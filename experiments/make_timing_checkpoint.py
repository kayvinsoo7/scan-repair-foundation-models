"""Random-weight TabPFN-2.5 checkpoint with the real architecture (default TabPFN-2.5 config).
Compute cost = real model; predictions are meaningless. Used ONLY for timing.
Usage: python make_timing_checkpoint.py   then   TABPFN_MODEL_PATH=<file> TABPFN_DEVICE=cpu python timing_tabpfn.py"""
import os, torch, dataclasses
from tabpfn.architectures import tabpfn_v2_5 as A
cfg = A.TabPFNV2p5Config(max_num_classes=10); model = A.get_architecture(cfg)
n = sum(p.numel() for p in model.parameters())
torch.save({"state_dict": model.state_dict(), "config": dataclasses.asdict(cfg), "architecture_name": "tabpfn_v2_5"}, os.environ.get("CKPT", "timing_tabpfn_v2_5_random_weights.ckpt"))
print("params:", f"{n/1e6:.1f}M", "| emsize", cfg.emsize, "layers", cfg.nlayers, "heads", cfg.nhead)
