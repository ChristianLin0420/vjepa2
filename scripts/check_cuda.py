"""Report whether this process can safely launch JEPA-4D CUDA workloads.

The checker intentionally uses only PyTorch and the Python standard library so
that it can run before the rest of the project is imported.  PCI devices are
discovered from ``nvidia-smi``/``lspci``; no host-specific bus address is
assumed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def command_result(command: list[str], timeout: int = 15) -> dict[str, Any]:
    """Run a diagnostic command without allowing it to abort the report."""

    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {"command": command, "returncode": None, "stdout": "", "stderr": f"unavailable: {error}"}


def _gpu_pci_inventory() -> tuple[dict[str, Any], list[str]]:
    smi = command_result(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,driver_version,memory.total,temperature.gpu,pstate",
            "--format=csv,noheader,nounits",
        ]
    )
    # 10de is NVIDIA's PCI vendor ID.  Querying by vendor discovers every
    # NVIDIA function without coupling the checker to a particular bus slot.
    lspci = command_result(["lspci", "-D", "-nn", "-d", "10de:"])
    pci_lines = str(lspci["stdout"]).splitlines()
    return {"nvidia_smi": smi, "lspci": {**lspci, "stdout": "\n".join(pci_lines)}}, pci_lines


def _visible_device_token(device_index: int) -> str | None:
    visible = os.getenv("CUDA_VISIBLE_DEVICES")
    if not visible:
        return str(device_index)
    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    return tokens[device_index] if device_index < len(tokens) else None


def _stress_cuda(torch: Any, device: Any, seconds: float, matrix_size: int, allocation_mib: int) -> dict[str, Any]:
    if seconds < 0:
        raise ValueError("--stress-seconds must be non-negative")
    if matrix_size < 256:
        raise ValueError("--matrix-size must be at least 256")
    if allocation_mib < 0:
        raise ValueError("--allocation-mib must be non-negative")

    torch.cuda.reset_peak_memory_stats(device)
    reserve = None
    if allocation_mib:
        elements = allocation_mib * 1024 * 1024 // torch.empty((), dtype=torch.float32).element_size()
        reserve = torch.empty(elements, dtype=torch.float32, device=device)
        reserve.fill_(1.0)
        torch.cuda.synchronize(device)

    if seconds == 0:
        return {
            "requested_seconds": seconds,
            "elapsed_seconds": 0.0,
            "iterations": 0,
            "matrix_size": matrix_size,
            "allocation_mib": allocation_mib,
            "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "finite": True,
        }

    first = torch.randn((matrix_size, matrix_size), device=device)
    second = torch.randn((matrix_size, matrix_size), device=device)
    # Warm up kernels before starting the sustained interval.
    value = first @ second
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    iterations = 0
    while time.perf_counter() - started < seconds:
        value = first @ second
        iterations += 1
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    checksum = float(value.float().mean().item())
    finite = bool(torch.isfinite(value).all().item())
    # A square GEMM performs approximately 2*N^3 floating-point operations.
    tflops = (iterations * 2.0 * matrix_size**3) / max(elapsed, 1e-9) / 1e12
    result = {
        "requested_seconds": seconds,
        "elapsed_seconds": elapsed,
        "iterations": iterations,
        "matrix_size": matrix_size,
        "allocation_mib": allocation_mib,
        "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "checksum": checksum,
        "finite": finite,
        "estimated_tflops": tflops,
    }
    del reserve, first, second, value
    torch.cuda.empty_cache()
    if not finite:
        raise RuntimeError("CUDA stress result contains non-finite values")
    return result


def _write_report(payload: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(rendered)
        temporary.replace(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0, help="Logical CUDA device index visible to this process.")
    parser.add_argument(
        "--stress-seconds",
        type=float,
        default=0.0,
        help="Run repeated CUDA matrix multiplications for at least this many seconds.",
    )
    parser.add_argument("--matrix-size", type=int, default=4096, help="Square matrix size used by the stress test.")
    parser.add_argument(
        "--allocation-mib",
        type=int,
        default=0,
        help="Keep this many MiB allocated throughout the stress test.",
    )
    parser.add_argument("--json-output", type=Path, help="Also atomically write the JSON report to this path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload: dict[str, Any] = {
        "schema_version": "jepa4d-cuda-health-v2",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "pid": os.getpid(),
        "slurm": {
            "job_id": os.getenv("SLURM_JOB_ID"),
            "job_name": os.getenv("SLURM_JOB_NAME"),
            "node_list": os.getenv("SLURM_JOB_NODELIST"),
            "local_id": os.getenv("SLURM_LOCALID"),
        },
        "requested_device": args.device,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "visible_device_token": _visible_device_token(args.device),
    }
    inventory, pci_lines = _gpu_pci_inventory()
    payload["system_inventory"] = inventory
    errors: list[str] = []

    try:
        import torch

        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count())
        payload.update(
            {
                "torch": torch.__version__,
                "torch_cuda_build": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "cuda_available": available,
                "device_count": count,
            }
        )
        if not available:
            errors.append("torch.cuda.is_available() is false")
        elif args.device < 0 or args.device >= count:
            errors.append(f"logical CUDA device {args.device} is outside the visible range [0, {count})")
        else:
            device = torch.device(f"cuda:{args.device}")
            torch.cuda.set_device(device)
            properties = torch.cuda.get_device_properties(device)
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            payload["selected_device"] = {
                "logical_index": args.device,
                "name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "total_memory_gib": properties.total_memory / 2**30,
                "free_memory_gib": free_bytes / 2**30,
                "runtime_total_memory_gib": total_bytes / 2**30,
                "multiprocessors": properties.multi_processor_count,
                "uuid": str(getattr(properties, "uuid", "")) or None,
            }
            try:
                payload["stress"] = _stress_cuda(
                    torch, device, args.stress_seconds, args.matrix_size, args.allocation_mib
                )
            except Exception as error:  # CUDA errors must be included in the persisted report.
                errors.append(f"CUDA stress failed: {type(error).__name__}: {error}")
    except Exception as error:
        payload["torch_import_error"] = f"{type(error).__name__}: {error}"
        errors.append("PyTorch could not be imported")

    if inventory["nvidia_smi"]["returncode"] != 0:
        errors.append("nvidia-smi inventory query failed")
    failed_pci = [line for line in pci_lines if "(rev ff)" in line.lower()]
    if failed_pci:
        errors.append(f"PCI configuration space is unreadable for {len(failed_pci)} GPU device(s) (revision ff)")

    payload["errors"] = errors
    payload["status"] = "pass" if not errors else "fail"
    if errors:
        payload["action"] = (
            "Do not launch model execution. Inspect the reported CUDA/NVML/PCI state on the allocated node; "
            "a revision-ff device or an NVML Unknown Error is a host/platform fault, not a Python package fault."
        )
    _write_report(payload, args.json_output)
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
