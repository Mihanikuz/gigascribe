"""Fail fast with an actionable error when the requested CUDA runtime is unsafe."""
from __future__ import annotations

import argparse
import sys


def smoke_test(require_cuda: bool = False) -> dict:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"PyTorch import failed: {exc}") from exc

    result = {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "available": torch.cuda.is_available(), "arch_list": torch.cuda.get_arch_list(),
    }
    if require_cuda and not result["available"]:
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false. Check NVIDIA Container Toolkit and driver visibility.")
    if not result["available"]:
        return result
    result["device_name"] = torch.cuda.get_device_name(0)
    result["capability"] = torch.cuda.get_device_capability(0)
    is_blackwell = result["capability"] == (12, 0)
    # Check actual capabilities (compiled arch list, CUDA runtime version),
    # not one exact torch version string: that broke every time the pinned
    # torch version changed even though newer compatible builds work fine.
    if is_blackwell and "sm_120" not in result["arch_list"]:
        raise RuntimeError(f"RTX 50xx (sm_120) needs a torch build with sm_120 support; installed torch {torch.__version__} (CUDA {torch.version.cuda}) does not include it.")
    if is_blackwell and torch.version.cuda:
        cuda_major_minor = tuple(int(p) for p in torch.version.cuda.split(".")[:2])
        if cuda_major_minor < (12, 8):
            raise RuntimeError(f"RTX 50xx (sm_120) needs CUDA 12.8 or newer; installed torch {torch.__version__} reports CUDA {torch.version.cuda}.")
    x = torch.ones((32, 32), device="cuda")
    torch.cuda.synchronize()
    result["smoke"] = float((x @ x).sum().item())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    try:
        print(smoke_test(args.require_cuda))
    except RuntimeError as exc:
        print(f"GPU smoke test failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
