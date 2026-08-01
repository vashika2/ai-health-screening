import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
from utils.gradcam import GradCAM, overlay_heatmap

class TBDetector:
    def __init__(self, weights_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"TB Model running on: {self.device}")

        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(2048, 2)

        self.class_to_idx = {'Normal': 0, 'TB': 1}

        if weights_path and os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.class_to_idx = checkpoint.get('class_to_idx', self.class_to_idx)
                print(f"TB weights loaded! Accuracy: {checkpoint.get('best_acc', 0):.2f}%")
            else:
                self.model.load_state_dict(checkpoint)
                print("TB weights loaded successfully")
            print(f"Class mapping: {self.class_to_idx}")
        else:
            print("No trained weights yet — using pretrained backbone only")

        self.model.eval().to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # ResNet-50's last conv layer — best for Grad-CAM
        self.target_layer = self.model.layer4[-1]
        self.gradcam = GradCAM(self.model, self.target_layer)

    def predict(self, image_path: str) -> dict:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)

        tb_idx = self.class_to_idx.get('TB', 1)
        normal_idx = self.class_to_idx.get('Normal', 0)

        tb_prob = probs[0][tb_idx].item()
        normal_prob = probs[0][normal_idx].item()

        label = "TB Detected" if tb_prob > 0.5 else "Normal"

        return {
            "label": label,
            "confidence": round(tb_prob * 100, 2),
            "normal_confidence": round(normal_prob * 100, 2),
            "severity": self._severity(tb_prob),
            "needs_referral": tb_prob > 0.7,
            "predicted_class_idx": tb_idx if tb_prob > 0.5 else normal_idx
        }

    def generate_heatmap(self, image_path: str, save_path: str, class_idx: int) -> str:
        """
        Generates a Grad-CAM heatmap showing which regions of the
        X-ray influenced the model's decision the most.
        """
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        tensor.requires_grad = True

        cam = self.gradcam.generate(tensor, class_idx)
        overlay_heatmap(image_path, cam, save_path)
        return save_path

    def _severity(self, prob: float) -> str:
        if prob < 0.5:    return "None"
        elif prob < 0.65: return "Low Suspicion"
        elif prob < 0.80: return "Moderate Suspicion"
        else:             return "High Suspicion"