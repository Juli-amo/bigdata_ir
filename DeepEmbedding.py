# DeepEmbedding.py
# Minimal, production-ready embedding wrapper around MobileNetV3 → 128D
from __future__ import annotations

import logging
from typing import Optional, List

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as T
    _TORCH_OK = True
except Exception as e:  # pragma: no cover
    _TORCH_OK = False
    _TORCH_ERR = e

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # optional dependency for embed_path()

log = logging.getLogger(__name__)


class VisionEmbedder:
    """MobileNetV3 backbone + 128D projection head.

    Loads a MobileNetV3 (large/small) backbone, replaces classifier with Identity,
    and adds a 128D linear projection + LayerNorm. Embeddings are L2-normalized.

    Typical use:
        emb = VisionEmbedder()
        z = emb.embed_path("image.jpg")  # -> np.ndarray shape (128,)
    """

    def __init__(
        self,
        model_name: str = "mobilenet_v3_large",
        proj_dim: int = 128,
        seed: int = 42,
    ) -> None:
        """Init model, device, preprocess, and 128D head."""
        if not _TORCH_OK:
            raise RuntimeError(f"PyTorch/torchvision not available: {_TORCH_ERR}")

        # Device preference: Apple MPS > CUDA > CPU
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        torch.manual_seed(seed)

        # Build backbone (try pretrained; fall back to random if weights missing/offline)
        self.backbone, feat_dim = self._build_backbone(model_name)
        self.backbone.to(self.device).eval()

        # 128D projection head
        self.head = nn.Sequential(
            nn.Linear(feat_dim, proj_dim, bias=False),
            nn.LayerNorm(proj_dim),
        ).to(self.device).eval()

        # Preprocessing (HWC uint8 -> CHW float, resize/crop/normalize)
        self.tf = T.Compose(
            [
                T.ToTensor(),
                T.Resize(256, antialias=True),
                T.CenterCrop(224),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self._available = True
        log.info(
            "VisionEmbedder initialized: %s on %s (feat_dim=%d, proj=%d)",
            model_name,
            self.device,
            feat_dim,
            proj_dim,
        )

    # ----------------------- internal helpers -----------------------

    def _build_backbone(self, model_name: str):
        """Create MobileNetV3 backbone; try pretrained weights first."""
        if model_name not in {"mobilenet_v3_large", "mobilenet_v3_small"}:
            model_name = "mobilenet_v3_large"

        try:
            if model_name == "mobilenet_v3_small":
                weights = models.MobileNet_V3_Small_Weights.DEFAULT
                net = models.mobilenet_v3_small(weights=weights)
            else:
                weights = models.MobileNet_V3_Large_Weights.DEFAULT
                net = models.mobilenet_v3_large(weights=weights)
            log.info("Loaded pretrained weights for %s.", model_name)
        except Exception as e:
            # No internet or incompatible torchvision version -> fallback
            log.warning(
                "Could not load pretrained weights for %s (%s). Falling back to random init.",
                model_name,
                e,
            )
            if model_name == "mobilenet_v3_small":
                net = models.mobilenet_v3_small(weights=None)
            else:
                net = models.mobilenet_v3_large(weights=None)

        # Replace classifier with Identity and read feature dimension
        if isinstance(net.classifier, nn.Sequential) and len(net.classifier) > 0:
            # First Linear holds in_features in official torchvision builds
            feat_dim = getattr(net.classifier[0], "in_features", None)
        else:
            feat_dim = None

        if feat_dim is None:
            # Fallback: infer by running a single dummy forward
            net.classifier = nn.Identity()
            with torch.inference_mode():
                dummy = torch.zeros(1, 3, 224, 224)
                y = net(dummy)
                feat_dim = y.shape[1]
        else:
            net.classifier = nn.Identity()

        return net, int(feat_dim)

    def _preprocess(self, img_bgr: np.ndarray) -> torch.Tensor:
        """BGR uint8 -> normalized tensor [1,3,224,224] on target device."""
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Empty image array.")
        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 BGR, got shape {img_bgr.shape}.")
        # Convert BGR -> RGB (copy to ensure contiguous memory)
        rgb = img_bgr[..., ::-1].copy()
        x = self.tf(rgb).unsqueeze(0).to(self.device)
        return x

    # ------------------------ public API ----------------------------

    def available(self) -> bool:
        """Return True if the embedder is usable."""
        return bool(self._available)

    @torch.inference_mode()
    def embed_array(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Embed a BGR image array -> float32 vector of shape (128,)."""
        if not self._available:
            return None
        x = self._preprocess(img_bgr)
        f = self.backbone(x)
        z = self.head(f)
        z = torch.nn.functional.normalize(z, p=2, dim=1)
        return z.squeeze(0).detach().cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def embed_path(self, path: str) -> Optional[np.ndarray]:
        """Read an image with cv2 and embed it (requires cv2)."""
        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) not available for embed_path().")
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            log.warning("embed_path: could not read image: %s", path)
            return None
        return self.embed_array(img)

    @torch.inference_mode()
    def embed_batch(self, imgs_bgr: List[np.ndarray], batch_size: int = 64) -> np.ndarray:
        """Embed a list of BGR images in mini-batches -> [N,128]."""
        if not imgs_bgr:
            return np.empty((0, 128), dtype=np.float32)
        outs: List[np.ndarray] = []
        for i in range(0, len(imgs_bgr), max(1, int(batch_size))):
            batch = imgs_bgr[i : i + batch_size]
            xs = [self._preprocess(im) for im in batch]
            x = torch.cat(xs, dim=0)  # [B,3,224,224]
            f = self.backbone(x)
            z = self.head(f)
            z = torch.nn.functional.normalize(z, p=2, dim=1)
            outs.append(z.detach().cpu().numpy().astype(np.float32))
        return np.vstack(outs)

    def close(self) -> None:
        """Free GPU caches if any; safe to call multiple times."""
        try:
            if self.device.type == "cuda":
                import torch.cuda as _tc  # lazy import
                _tc.empty_cache()
        except Exception:
            pass