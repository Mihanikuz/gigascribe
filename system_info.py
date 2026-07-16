from __future__ import annotations
import importlib, shutil, subprocess, time
from pathlib import Path
from typing import Any

def package_version(name: str) -> str | None:
    try:
        mod = importlib.import_module(name)
        return getattr(mod, '__version__', None)
    except Exception:
        return None

def run_cuda_test() -> dict[str, Any]:
    try:
        import torch
        if not torch.cuda.is_available():
            return {'ok': False, 'error': 'CUDA недоступна'}
        x = torch.randn((2048, 2048), device='cuda')
        _ = x @ x
        torch.cuda.synchronize()
        return {'ok': True, 'error': None}
    except Exception as exc:
        return {'ok': False, 'error': str(exc)}

def gpu_info(real_test: bool = False) -> dict[str, Any]:
    info: dict[str, Any] = {'mode': 'cpu', 'cuda_available': False, 'real_cuda_test': {'ok': False, 'error': 'not run'}}
    try:
        import torch
        arch_list = torch.cuda.get_arch_list()
        info.update({'torch_version': torch.__version__, 'cuda_runtime': torch.version.cuda, 'arch_list': arch_list})
        if torch.cuda.is_available():
            free = total = None
            try: free, total = torch.cuda.mem_get_info(0)
            except Exception: pass
            props = torch.cuda.get_device_properties(0)
            info.update({'mode':'cuda','cuda_available':True,'gpu_name':torch.cuda.get_device_name(0),'compute_capability': f'{props.major}.{props.minor}', 'total_vram': total or props.total_memory, 'free_vram': free, 'used_vram': (total-free) if free is not None and total is not None else None, 'has_sm_120':'sm_120' in arch_list})
            if (props.major, props.minor) == (12, 0) and 'sm_120' not in arch_list:
                info['compatibility_error'] = 'RTX 50xx detected but this PyTorch wheel lacks sm_120; rebuild with requirements-cu128.txt.'
    except Exception as exc:
        info['torch_error'] = str(exc)
    info['torchaudio_version'] = package_version('torchaudio')
    if shutil.which('nvidia-smi'):
        try:
            r = subprocess.run(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'], text=True, capture_output=True, timeout=5)
            info['nvidia_driver'] = r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else None
        except Exception as exc: info['nvidia_smi_error'] = str(exc)
    if real_test: info['real_cuda_test'] = run_cuda_test()
    return info
