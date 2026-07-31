"""Grad-CAM on the final convolutional block of HydroSense-Base/SE (README §11.1).

Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017):
back-propagate the target class score to the last conv block's feature
maps, weight each channel by its average gradient, and take a ReLU'd
weighted sum — producing a (freq, time) heatmap over the input spectrogram
that highlights which time-frequency regions drove the prediction.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Hooks `target_layer`'s forward activations and backward gradients.

    Usage:
        cam = GradCAM(model, model.target_layer)
        heatmap = cam(input_tensor, class_idx=predicted_class)  # (H, W) in [0, 1]
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        self._fwd_handle = target_layer.register_forward_hook(self._save_activations)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, output):  # noqa: ARG002
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):  # noqa: ARG002
        self._gradients = grad_output[0].detach()

    def remove(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, int]:
        """Returns (heatmap resized to x's spatial shape, the class index explained)."""
        self.model.eval()
        was_training = self.model.training
        logits = self.model(x)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1)[0].item())

        self.model.zero_grad(set_to_none=True)
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=False)

        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks did not fire; check that `target_layer` is on the forward path."
            )

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = F.relu((weights * self._activations).sum(dim=1, keepdim=True))  # (B, 1, h, w)

        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        self.model.train(was_training)
        return cam, class_idx


def overlay_heatmap(
    spectrogram: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    """Blend a Grad-CAM heatmap over a normalised (freq, time) spectrogram for visualisation.

    Returns an RGB array in [0, 1] suitable for `matplotlib.pyplot.imshow`.
    """
    import matplotlib

    def _cmap(name: str):
        try:
            return matplotlib.colormaps[name]
        except (AttributeError, KeyError):
            import matplotlib.cm as cm

            return cm.get_cmap(name)

    spec_norm = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min() + 1e-8)
    base_rgb = _cmap("gray")(spec_norm)[..., :3]
    heat_rgb = _cmap("jet")(heatmap)[..., :3]
    return np.clip((1 - alpha) * base_rgb + alpha * heat_rgb, 0, 1)
