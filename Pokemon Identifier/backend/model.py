import json
import torch
import timm
from PIL import Image
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor

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
        predictions = self.predict_topk(img_path, k=1)
        top = predictions[0]
        return top["name"], top["confidence"]

    def predict_topk(self, img_path, k=3):
        image = Image.open(img_path).convert("RGB")
        x = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            feats = self.model.forward_features(x)
            logits = self.model.forward_head(feats)
            probs = torch.softmax(logits, dim=1)[0]
            values, indices = torch.topk(probs, k=min(k, probs.shape[0]))

        results = []
        for score, idx in zip(values.tolist(), indices.tolist()):
            results.append({
                "name": self.idx_to_class[idx].capitalize(),
                "confidence": float(score)
            })
        return results

class PokemonPrecheck:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = "cpu"
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.labels = [
            "a photo of a pokemon character",
            "a photo of a cartoon monster",
            "a photo of a real animal",
            "a photo of a person",
            "a photo of a car",
            "a photo of a landscape"
        ]

    def predict(self, img_path):
        image = Image.open(img_path).convert("RGB")
        inputs = self.processor(text=self.labels, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]

        scores = probs.tolist()
        labeled = [{"label": label, "confidence": float(score)} for label, score in zip(self.labels, scores)]
        labeled_sorted = sorted(labeled, key=lambda x: x["confidence"], reverse=True)
        pokemon_score = labeled_sorted[0]["confidence"] if labeled_sorted[0]["label"] in self.labels[:2] else scores[0]
        is_pokemon = pokemon_score >= 0.4
        return {
            "is_pokemon": is_pokemon,
            "pokemon_score": float(pokemon_score),
            "top_labels": labeled_sorted[:3]
        }
