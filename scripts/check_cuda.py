"""Report whether this process can safely launch JEPA-4D CUDA workloads."""

from __future__ import annotations

import json
import subprocess

import torch


def command_output(command: list[str]) -> str:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
        return (result.stdout or result.stderr).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return f"unavailable: {error}"


def main() -> None:
    available = torch.cuda.is_available()
    payload: dict[str, object] = {
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": available,
        "device_count": torch.cuda.device_count(),
        "nvidia_smi": command_output(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv"]
        ),
        "pci_device": command_output(["lspci", "-s", "65:00.0", "-nn"]),
    }
    if available:
        properties = torch.cuda.get_device_properties(0)
        payload.update(
            {
                "device_name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "total_memory_gb": properties.total_memory / 2**30,
            }
        )
    else:
        payload["action"] = (
            "CUDA is unavailable. If lspci reports revision ff or nvidia-smi reports Unknown Error, the host must "
            "restore/reset the PCI device before model execution; changing Python packages cannot repair that state."
        )
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if available else 1)


if __name__ == "__main__":
    main()
