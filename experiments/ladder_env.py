"""Device and software description recorded with every measurement."""
import platform
import tabpfn_common as TC
TAG = "v1" if TC.BACKEND == "v1" else TC.VERSION

def device_name():
    import torch
    if TC.BACKEND == "v1": return "cpu"
    if torch.cuda.is_available() and TC.DEVICE in ("auto", "cuda"): return "cuda:" + torch.cuda.get_device_name(0)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() and TC.DEVICE in ("auto", "mps"): return "mps"
    return "cpu:" + (platform.processor() or platform.machine())

def env_info():
    import torch
    info = dict(backend=TC.BACKEND, version=TAG, device=device_name(), torch=torch.__version__, python=platform.python_version(),
                cpu_threads=torch.get_num_threads(), machine=platform.platform())
    try:
        import tabpfn; info["tabpfn"] = getattr(tabpfn, "__version__", "?")
    except Exception: pass
    return info
