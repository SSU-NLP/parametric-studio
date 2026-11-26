import torch

_DTYPE = {"cuda": torch.bfloat16, "mps": torch.bfloat16, "cpu": torch.float32}


def resolve_device(name="auto"):
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")  # ponytail: graceful fallback; script's resolve_device SystemExits here
    return torch.device(name)  # "cuda:1" etc. pass through when cuda is present


def pick_device(device=None):
    """Resolve an explicit device, or auto-place on the CUDA card with the most free memory.

    device given ("cuda:1", "cpu", ...) → resolve_device as-is. None/"auto" with several CUDA
    cards → the one with max free memory (so a second model lands on the idle card). mps/cpu/single
    cuda → whatever resolve_device("auto") picks."""
    if device not in (None, "auto"):
        return resolve_device(device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
        return torch.device(f"cuda:{free.index(max(free))}")
    return resolve_device("auto")


def dtype_for(device):
    return _DTYPE.get(torch.device(device).type, torch.float32)
