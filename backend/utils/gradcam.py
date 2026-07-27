import torch
import numpy as np
import cv2
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.eval()
        output = self.model(input_tensor)

        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = torch.relu(cam)

        cam = cam.cpu().numpy()

        # Normalize using percentile clipping — makes the hottest
        # regions stand out more clearly instead of a flat gradient
        if cam.max() > cam.min():
            lower = np.percentile(cam, 40)
            cam = np.clip(cam, lower, None)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)

        return cam


def overlay_heatmap(original_image_path: str, cam: np.ndarray, save_path: str, alpha: float = 0.45):
    """
    Overlays the Grad-CAM heatmap on top of the ORIGINAL resolution image
    so the X-ray/retina remains clearly visible underneath the color map.
    """
    # Load original image at its real size and ensure 3-channel BGR
    img = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image at {original_image_path}")

    h, w = img.shape[:2]

    # Resize the CAM (small, e.g. 7x7 or 12x12) UP to the image's real size
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    cam_resized = np.clip(cam_resized, 0, 1)

    # Apply colormap (JET: blue=low, red=high)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = heatmap.astype(np.float32)

    img_float = img.astype(np.float32)

    # Blend: keep the base X-ray clearly visible, heatmap as a translucent overlay
    overlayed = img_float * (1 - alpha) + heatmap * alpha
    overlayed = np.clip(overlayed, 0, 255).astype(np.uint8)

    cv2.imwrite(save_path, overlayed)
    return save_path