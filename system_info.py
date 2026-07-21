from __future__ import annotations
from typing import Any

def gpu_info(real_test: bool = False) -> dict[str, Any]:
    info: dict[str, Any] = {"mode": "cuda", "cuda_available": False, "real_cuda_test": {"ok": False, "error": "not run"}}
    try:
        import torch
        info.update({"pytorch": torch.__version__, "cuda_runtime": torch.version.cuda, "cuda_available": bool(torch.cuda.is_available())})
        if torch.cuda.is_available():
            props=torch.cuda.get_device_properties(0)
            info.update({"device_name": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0), "total_vram_gb": props.total_memory/(1024**3)})
            if real_test:
                x=torch.ones((32,32), device="cuda"); y=x@x; torch.cuda.synchronize()
                info["real_cuda_test"]={"ok": bool(y[0,0].item() == 32.0)}
    except Exception as exc:
        info["real_cuda_test"]={"ok": False, "error": str(exc)}
    return info
