import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
from utils.gradcam import GradCAM, overlay_heatmap

class DRGrader:
    def __init__(self, weights_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DR Model running on: {self.device}")

        self.model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 5)

        self.class_to_idx = {}

        if weights_path and os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.class_to_idx = checkpoint.get('class_to_idx', {})
                print(f"DR weights loaded! Accuracy: {checkpoint.get('best_acc', 0):.2f}%")
            else:
                self.model.load_state_dict(checkpoint)
                print("DR weights loaded successfully")
            print(f"Class mapping: {self.class_to_idx}")
        else:
            print("No trained weights yet — using pretrained backbone only")

        self.model.eval().to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.CenterCrop(380),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.grade_labels = {
            0: "No DR", 1: "Mild DR", 2: "Moderate DR",
            3: "Severe DR", 4: "Proliferative DR"
        }

        self.folder_to_grade = {
            "No_DR": 0, "Mild": 1, "Moderate": 2,
            "Severe": 3, "Proliferate_DR": 4
        }

        # EfficientNet's last conv block — best for Grad-CAM
        self.target_layer = self.model.features[-1]
        self.gradcam = GradCAM(self.model, self.target_layer)

    def predict(self, image_path: str) -> dict:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)

        if self.class_to_idx:
            idx_to_grade = {}
            for folder_name, model_idx in self.class_to_idx.items():
                grade = self.folder_to_grade.get(folder_name, model_idx)
                idx_to_grade[model_idx] = grade

            predicted_idx = torch.argmax(probs, dim=1).item()
            grade = idx_to_grade.get(predicted_idx, predicted_idx)
            confidence = probs[0][predicted_idx].item()

            all_grades = {}
            for folder_name, model_idx in self.class_to_idx.items():
                g = self.folder_to_grade.get(folder_name, model_idx)
                all_grades[self.grade_labels[g]] = round(probs[0][model_idx].item() * 100, 2)
        else:
            predicted_idx = torch.argmax(probs, dim=1).item()
            grade = predicted_idx
            confidence = probs[0][predicted_idx].item()
            all_grades = {
                self.grade_labels[i]: round(probs[0][i].item() * 100, 2)
                for i in range(5)
            }

        return {
            "grade": grade,
            "label": self.grade_labels.get(grade, "Unknown"),
            "confidence": round(confidence * 100, 2),
            "needs_referral": grade >= 3,
            "all_grades": all_grades,
            "predicted_class_idx": predicted_idx
        }

    def generate_heatmap(self, image_path: str, save_path: str, class_idx: int) -> str:
        """
        Generates a Grad-CAM heatmap showing which regions of the
        retina influenced the model's decision the most.
        """
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        tensor.requires_grad = True

        cam = self.gradcam.generate(tensor, class_idx)
        overlay_heatmap(image_path, cam, save_path)
        return save_path