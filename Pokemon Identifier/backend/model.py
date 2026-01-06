import json
import torch
import timm
from PIL import Image
from torchvision import transforms

class PokemonModel:
    def __init__(self, model_dir="models"):
        self.device = "cpu"

        # Load config
        with open(f"{model_dir}/inference_config.json") as f:
            cfg = json.load(f)

        # Load class map
        with open(f"{model_dir}/class_map.json") as f:
            class_map = json.load(f)
        self.idx_to_class = {v: k for k, v in class_map.items()}

        # Load model
        self.model = timm.create_model(
            cfg["model_name"],
            pretrained=False,
            num_classes=cfg["num_classes"]
        )
        self.model.load_state_dict(
            torch.load(f"{model_dir}/model.pth", map_location="cpu")
        )
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((cfg["img_size"], cfg["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(cfg["mean"], cfg["std"])
        ])

    def predict(self, img_path):
        image = Image.open(img_path).convert("RGB")
        x = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            feats = self.model.forward_features(x)
            logits = self.model.forward_head(feats)
            probs = torch.softmax(logits, dim=1)
            idx = probs.argmax(dim=1).item()

        return self.idx_to_class[idx], float(probs[0, idx])
