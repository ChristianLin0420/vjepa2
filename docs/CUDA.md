# CUDA environment and recovery

## Supported project environment

The current host uses NVIDIA driver `535.309.01`. Install the project CUDA environment with:

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pip install -r requirements-cuda.txt
```

`requirements-cuda.txt` constrains PyTorch to `2.7.1+cu118` and torchvision to `0.22.1+cu118`. CUDA 11.8 is compatible
with the installed driver. Do not install an unconstrained nightly/PyPI PyTorch build: it may select CUDA 13 and fail
with `The NVIDIA driver on your system is too old` even when the GPU is healthy.

Verify both discovery and a real kernel launch:

```bash
nvidia-smi
.venv/bin/python - <<'PY'
import torch

assert torch.cuda.is_available()
x = torch.arange(8, device="cuda")
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), (x * x).tolist())
PY
```

## `Xid 79`: GPU fallen off the bus

If `lspci` reports the GPU as revision `ff`, `nvidia-smi` cannot determine a device handle, and the kernel log contains
`Xid 79, GPU has fallen off the bus`, the device is not responding over PCIe. Reinstalling Python packages cannot repair
that hardware state. Preserve diagnostics before recovery:

```bash
sudo nvidia-bug-report.sh --safe-mode
sudo dmesg -T | grep -E 'NVRM|Xid|fallen off the bus'
```

Try `sudo nvidia-smi --gpu-reset -i 0` only when no process owns the device. If the reset cannot obtain a device handle,
or a PCI function-level reset reports that the device is not ready, reboot the host. If revision `ff` remains after a
warm reboot, perform a cold power cycle and inspect the PCIe slot, auxiliary power, thermals, and upstream switch/root
port. Repeated Xid 79 failures require host/hardware remediation rather than application changes.

After recovery, run the kernel-launch check above and the CUDA-only model tests:

```bash
.venv/bin/pytest tests/models/test_vision_transformer.py -q
```
