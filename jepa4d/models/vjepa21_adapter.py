"""V-JEPA 2.1 dense feature extraction with deterministic CPU mocks."""

from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from jepa4d.data.schemas import JEPATokenBundle, RGBInputBatch
from jepa4d.data.transforms import resize_center_crop_normalize

ModelName = Literal[
    "vjepa2_1_vit_base_384",
    "vjepa2_1_vit_large_384",
    "vjepa2_1_vit_giant_384",
    "vjepa2_1_vit_gigantic_384",
]

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "vjepa2_1_vit_base_384": {"embed_dim": 768, "layers": (2, 5, 8, 11), "tubelet": 2},
    "vjepa2_1_vit_large_384": {"embed_dim": 1024, "layers": (5, 11, 17, 23), "tubelet": 2},
    "vjepa2_1_vit_giant_384": {"embed_dim": 1408, "layers": (9, 19, 29, 39), "tubelet": 2},
    "vjepa2_1_vit_gigantic_384": {"embed_dim": 1664, "layers": (11, 23, 37, 47), "tubelet": 2},
}


class VJEPA21FeatureExtractor(nn.Module):
    """Expose V-JEPA 2.1 features in a stable view/time-aware contract.

    ``mock=True`` never downloads weights and is deterministic on CPU. Real mode
    supports official native ``.pt`` checkpoints and the local Hugging Face
    conversion used by this repository's Phase 1 smoke experiment.
    """

    def __init__(
        self,
        model_name: str = "vjepa2_1_vit_base_384",
        *,
        frozen: bool = True,
        mock: bool = False,
        checkpoint: str | Path | None = None,
        backend: Literal["auto", "native", "hf_compat"] = "auto",
        mock_embed_dim: int = 64,
        device: str | torch.device = "cpu",
        implementation_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        if model_name not in MODEL_SPECS:
            raise ValueError(f"unknown V-JEPA 2.1 model: {model_name}")
        self.model_name = model_name
        self.spec = MODEL_SPECS[model_name]
        self.frozen = frozen
        self.mock = mock
        self.checkpoint = None if checkpoint is None else Path(checkpoint)
        self.backend = backend
        self.device_name = str(device)
        self.implementation_path = None if implementation_path is None else Path(implementation_path)
        self.embed_dim = mock_embed_dim if mock else int(self.spec["embed_dim"])
        self.model: nn.Module | None = None
        self._loaded_backend = "mock" if mock else "unloaded"
        self._load_seconds = 0.0
        if not mock:
            self._load_real_model()

    @property
    def model_config(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "backend": self._loaded_backend,
            "frozen": self.frozen,
            "embed_dim": self.embed_dim,
            "patch_size": 16,
            "input_size": 384,
            "tubelet_size": self.spec["tubelet"],
            "intermediate_layers": list(self.spec["layers"]),
            "checkpoint": None if self.checkpoint is None else str(self.checkpoint),
            "load_seconds": self._load_seconds,
        }

    def _load_real_model(self) -> None:
        if self.checkpoint is None:
            raise ValueError("real mode requires a local checkpoint; run scripts/download_checkpoints.py")
        backend = self.backend
        if backend == "auto":
            backend = "native" if self.checkpoint.suffix in {".pt", ".pth"} else "hf_compat"
        started = time.perf_counter()
        if backend == "native":
            self.model = self._load_native(self.checkpoint)
        else:
            self.model = self._load_hf_compat(self.checkpoint)
        self.model.to(self.device_name).eval()
        if self.frozen:
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
        self._loaded_backend = backend
        self._load_seconds = time.perf_counter() - started

    def _load_native(self, checkpoint: Path) -> nn.Module:
        from app.vjepa_2_1.models import vision_transformer

        constructor = {
            "vjepa2_1_vit_base_384": "vit_base",
            "vjepa2_1_vit_large_384": "vit_large",
            "vjepa2_1_vit_giant_384": "vit_giant_xformers",
            "vjepa2_1_vit_gigantic_384": "vit_gigantic_xformers",
        }[self.model_name]
        model = vision_transformer.__dict__[constructor](
            img_size=(384, 384),
            patch_size=16,
            num_frames=64,
            tubelet_size=2,
            use_sdpa=True,
            use_rope=True,
            img_temporal_dim_size=1,
            interpolate_rope=True,
            n_output_distillation=1 if "base" in self.model_name or "large" in self.model_name else 4,
        )
        raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
        key = "ema_encoder" if "base" in self.model_name or "large" in self.model_name else "target_encoder"
        state = raw[key]
        state = {name.replace("module.", "").replace("backbone.", ""): value for name, value in state.items()}
        model.load_state_dict(state, strict=True)
        return model

    def _load_hf_compat(self, checkpoint: Path) -> nn.Module:
        root = checkpoint if checkpoint.is_dir() else checkpoint.parent
        impl = self.implementation_path or root.parent / "vjepa21_hf_impl"
        if not (impl / "modeling_vjepa21.py").exists():
            raise FileNotFoundError(
                f"V-JEPA 2.1 compatibility implementation not found at {impl}; "
                "run scripts/download_checkpoints.py --with-hf-implementation"
            )
        sys.path.insert(0, str(impl.parent.resolve()))
        package = impl.name
        configuration = importlib.import_module(f"{package}.configuration_vjepa21")
        modeling = importlib.import_module(f"{package}.modeling_vjepa21")
        config_data = json.loads((root / "config.json").read_text())
        config = configuration.VJEPA21Config(
            patch_size=config_data["patch_size"],
            crop_size=config_data["crop_size"],
            frames_per_clip=config_data["frames_per_clip"],
            tubelet_size=config_data["tubelet_size"],
            hidden_size=config_data["hidden_size"],
            in_chans=config_data["in_chans"],
            num_attention_heads=config_data["num_attention_heads"],
            num_hidden_layers=config_data["num_hidden_layers"],
            drop_path_rate=config_data["drop_path_rate"],
            mlp_ratio=config_data["mlp_ratio"],
            layer_norm_eps=config_data["layer_norm_eps"],
            qkv_bias=config_data["qkv_bias"],
            hidden_act=config_data["hidden_act"],
            img_temporal_dim_size=1,
            interpolate_rope=config_data["interpolate_rope"],
            modality_embedding=config_data.get("use_modality_embeddings", True),
            n_output_distillation=config_data.get("num_distillation_outputs", 1),
        )
        encoder = modeling.VJEPA21Encoder(config)
        from safetensors.torch import load_file

        weights = load_file(root / "model.safetensors")
        state = {
            name.removeprefix("encoder."): value for name, value in weights.items() if name.startswith("encoder.")
        }
        # The base conversion stores its sole trained distillation norm under a
        # compact name; it corresponds to the final hierarchical block.
        if "distillation_norms.0.weight" in state:
            last_norm = len(config.encoder_hierarchical_layers) - 1
            state[f"norms_block.{last_norm}.weight"] = state.pop("distillation_norms.0.weight")
            state[f"norms_block.{last_norm}.bias"] = state.pop("distillation_norms.0.bias")
        message = encoder.load_state_dict(state, strict=False)
        used_missing = [name for name in message.missing_keys if not name.startswith("norms_block.")]
        if used_missing or message.unexpected_keys:
            raise RuntimeError(f"incompatible V-JEPA 2.1 encoder weights: {message}")
        return encoder

    def _mock_forward(self, images: torch.Tensor, output_steps: int) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        batch_views, input_steps, _, height, width = images.shape
        frame_features = images.mean(dim=2, keepdim=True).reshape(batch_views * input_steps, 1, height, width)
        pooled = F.adaptive_avg_pool2d(frame_features, (24, 24)).flatten(2).transpose(1, 2)
        basis = torch.linspace(0.5, 2.0, self.embed_dim, device=images.device, dtype=images.dtype)
        dense = torch.sin(pooled * basis.view(1, 1, -1))
        dense = dense.reshape(batch_views, input_steps, 576, self.embed_dim)
        if output_steps != input_steps:
            dense = dense[:, : output_steps * 2].reshape(batch_views, output_steps, 2, 576, self.embed_dim).mean(dim=2)
        layers = {
            int(layer): dense * ((index + 1) / len(self.spec["layers"]))
            for index, layer in enumerate(self.spec["layers"])
        }
        return dense, layers

    def forward(self, batch: RGBInputBatch) -> JEPATokenBundle:
        started = time.perf_counter()
        normalized = resize_center_crop_normalize(batch.images.to(self.device_name), size=384)
        batch_size, views, input_steps = normalized.shape[:3]
        is_image = input_steps == 1
        output_steps = 1 if is_image else (input_steps + 1) // 2
        valid_source = batch.valid_mask
        if not is_image and input_steps % 2:
            normalized = torch.cat((normalized, normalized[:, :, -1:]), dim=2)
            valid_source = torch.cat((valid_source, valid_source[:, :, -1:]), dim=2)
            input_steps += 1
        clips = normalized.reshape(batch_size * views, input_steps, 3, 384, 384)
        layer_tokens: dict[int, torch.Tensor] = {}
        if self.mock:
            dense, raw_layers = self._mock_forward(clips, output_steps)
            layer_tokens = {
                layer: value.reshape(batch_size, views, output_steps, 576, self.embed_dim)
                for layer, value in raw_layers.items()
            }
        else:
            assert self.model is not None
            model_input = clips.permute(0, 2, 1, 3, 4).contiguous()
            captured: dict[int, torch.Tensor] = {}
            blocks = getattr(self.model, "layer", getattr(self.model, "blocks", None))
            handles = []
            if blocks is not None:
                for layer in self.spec["layers"]:

                    def capture(
                        _module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: Any, index: int = layer
                    ) -> None:
                        captured[index] = output[0] if isinstance(output, tuple) else output

                    handles.append(blocks[layer].register_forward_hook(capture))
            context = torch.no_grad() if self.frozen else torch.enable_grad()
            try:
                with context:
                    result = self.model(model_input)
                    dense_flat = result.last_hidden_state if hasattr(result, "last_hidden_state") else result
            finally:
                for handle in handles:
                    handle.remove()
            dense = dense_flat.reshape(batch_size * views, output_steps, 576, self.embed_dim)
            layer_tokens = {
                layer: value.reshape(batch_size, views, output_steps, 576, self.embed_dim)
                for layer, value in captured.items()
            }
            final_layer = int(self.spec["layers"][-1])
            layer_tokens[final_layer] = dense.reshape(batch_size, views, output_steps, 576, self.embed_dim)
        dense = dense.reshape(batch_size, views, output_steps, 576, self.embed_dim)
        if is_image:
            valid = batch.valid_mask
        else:
            valid = valid_source[..., : output_steps * 2].reshape(batch_size, views, output_steps, 2).any(dim=-1)
        return JEPATokenBundle(
            dense_tokens=dense,
            global_tokens=dense.mean(dim=-2),
            layer_tokens=layer_tokens,
            patch_grid=(24, 24),
            feature_scale=16,
            modality="image" if is_image else "video",
            valid_mask=valid,
            metadata={
                "model": self.model_config,
                "input_mode": batch.mode,
                "input_steps": batch.images.shape[2],
                "output_temporal_bins": output_steps,
                "forward_seconds": time.perf_counter() - started,
            },
        )
