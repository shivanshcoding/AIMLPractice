import csv
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from model import PokemonModel, PokemonPrecheck

app = FastAPI()

cors_origins = os.getenv("CORS_ORIGINS", "*")
origins = ["*"] if cors_origins == "*" else [origin.strip() for origin in cors_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

pokemon_model = PokemonModel()
try:
    precheck_model = PokemonPrecheck()
    precheck_error = None
except Exception as exc:
    precheck_model = None
    precheck_error = str(exc)

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
CSV_PATH = DATASET_DIR / "updated_pokedex_dataset.csv"
IMAGES_DIR = DATASET_DIR / "Pokedex Image Dataset"

app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

POKEAPI_BASE = "https://pokeapi.co/api/v2"

@app.get("/")
def root():
    return {
        "status": "running",
        "service": "Pokemon Identifier API",
        "version": "1.0.0",
        "docs": "/docs"
    }

TYPE_CHART = {
    "normal": {
        "double": [],
        "half": ["rock", "steel"],
        "none": ["ghost"]
    },
    "fire": {
        "double": ["grass", "ice", "bug", "steel"],
        "half": ["fire", "water", "rock", "dragon"],
        "none": []
    },
    "water": {
        "double": ["fire", "ground", "rock"],
        "half": ["water", "grass", "dragon"],
        "none": []
    },
    "electric": {
        "double": ["water", "flying"],
        "half": ["electric", "grass", "dragon"],
        "none": ["ground"]
    },
    "grass": {
        "double": ["water", "ground", "rock"],
        "half": ["fire", "grass", "poison", "flying", "bug", "dragon", "steel"],
        "none": []
    },
    "ice": {
        "double": ["grass", "ground", "flying", "dragon"],
        "half": ["fire", "water", "ice", "steel"],
        "none": []
    },
    "fighting": {
        "double": ["normal", "ice", "rock", "dark", "steel"],
        "half": ["poison", "flying", "psychic", "bug", "fairy"],
        "none": ["ghost"]
    },
    "poison": {
        "double": ["grass", "fairy"],
        "half": ["poison", "ground", "rock", "ghost"],
        "none": ["steel"]
    },
    "ground": {
        "double": ["fire", "electric", "poison", "rock", "steel"],
        "half": ["grass", "bug"],
        "none": ["flying"]
    },
    "flying": {
        "double": ["grass", "fighting", "bug"],
        "half": ["electric", "rock", "steel"],
        "none": []
    },
    "psychic": {
        "double": ["fighting", "poison"],
        "half": ["psychic", "steel"],
        "none": ["dark"]
    },
    "bug": {
        "double": ["grass", "psychic", "dark"],
        "half": ["fire", "fighting", "poison", "flying", "ghost", "steel", "fairy"],
        "none": []
    },
    "rock": {
        "double": ["fire", "ice", "flying", "bug"],
        "half": ["fighting", "ground", "steel"],
        "none": []
    },
    "ghost": {
        "double": ["psychic", "ghost"],
        "half": ["dark"],
        "none": ["normal"]
    },
    "dragon": {
        "double": ["dragon"],
        "half": ["steel"],
        "none": ["fairy"]
    },
    "dark": {
        "double": ["psychic", "ghost"],
        "half": ["fighting", "dark", "fairy"],
        "none": []
    },
    "steel": {
        "double": ["ice", "rock", "fairy"],
        "half": ["fire", "water", "electric", "steel"],
        "none": []
    },
    "fairy": {
        "double": ["fighting", "dragon", "dark"],
        "half": ["fire", "poison", "steel"],
        "none": []
    }
}

def normalize_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())

def normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "-")

def image_candidates(name: str):
    base = name.lower().strip()
    underscore = re.sub(r"[^a-z0-9]+", "_", base).strip("_")
    dash = re.sub(r"[^a-z0-9]+", "-", base).strip("-")
    compact = re.sub(r"[^a-z0-9]+", "", base)
    return [f"{underscore}.png", f"{dash}.png", f"{compact}.png"]

@lru_cache(maxsize=2048)
def fetch_pokeapi_image(name: str):
    normalized = normalize_name(name)
    res = requests.get(f"{POKEAPI_BASE}/pokemon/{normalized}", timeout=15)
    if res.status_code != 200:
        return None
    data = res.json()
    return (
        data.get("sprites", {})
        .get("other", {})
        .get("official-artwork", {})
        .get("front_default")
        or data.get("sprites", {}).get("front_default")
    )

def load_dataset():
    if not CSV_PATH.exists():
        raise RuntimeError(f"Dataset CSV not found at {CSV_PATH}")

    dataset = {}
    parent_map = {}
    children_map = {}

    with open(CSV_PATH, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            name = row["Name"].strip()
            key = normalize_key(name)
            type2 = row["Type2"].strip() if row["Type2"] else ""
            evolution = row.get("Evolution", "").strip()
            legendary = row.get("Legendary", "").strip().lower() == "true"

            dataset[key] = {
                "num": int(row["Num"]),
                "name": name,
                "type1": row["Type1"].strip(),
                "type2": type2 if type2 else None,
                "stats": {
                    "hp": int(row["HP"]),
                    "attack": int(row["Attack"]),
                    "defense": int(row["Defense"]),
                    "spatk": int(row["SpAtk"]),
                    "spdef": int(row["SpDef"]),
                    "speed": int(row["Speed"])
                },
                "generation": int(row["Generation"]),
                "legendary": legendary,
                "evolution": evolution
            }

            if evolution:
                parent_key = normalize_key(evolution)
                parent_map[key] = evolution
                children_map.setdefault(parent_key, []).append(name)

    return dataset, parent_map, children_map

DATASET, PARENT_MAP, CHILDREN_MAP = load_dataset()

def build_evolution_chain(name: str):
    chain = []
    current = name
    while current:
        chain.insert(0, current)
        parent = PARENT_MAP.get(normalize_key(current))
        if parent:
            current = parent
        else:
            break
    children = CHILDREN_MAP.get(normalize_key(name), [])
    for child in children:
        chain.append(child)
    return chain

@lru_cache(maxsize=2048)
def fetch_pokemon_data(name: str):
    key = normalize_key(name)
    data = DATASET.get(key)
    if not data:
        raise HTTPException(status_code=404, detail="Pokemon not found")
    image_url = fetch_pokeapi_image(data["name"])
    if not image_url:
        for candidate in image_candidates(data["name"]):
            image_path = IMAGES_DIR / candidate
            if image_path.exists():
                image_url = f"/images/{candidate}"
                break
    return {
        "num": data["num"],
        "name": data["name"],
        "type1": data["type1"],
        "type2": data["type2"],
        "stats": data["stats"],
        "generation": data["generation"],
        "legendary": data["legendary"],
        "evolution_chain": build_evolution_chain(data["name"]),
        "image_url": image_url
    }

@lru_cache(maxsize=128)
def fetch_type_effectiveness(type_name: str):
    normalized = normalize_name(type_name)
    chart = TYPE_CHART.get(normalized)
    if not chart:
        raise HTTPException(status_code=404, detail="Type not found")
    return {
        "double_damage_to": [t.capitalize() for t in chart["double"]],
        "half_damage_to": [t.capitalize() for t in chart["half"]],
        "no_damage_to": [t.capitalize() for t in chart["none"]]
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    suffix = os.path.splitext(file.filename or "")[1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        temp_file.write(content)
        temp_path = temp_file.name

    try:
        if precheck_model:
            try:
                precheck = precheck_model.predict(temp_path)
            except Exception:
                precheck = {
                    "is_pokemon": True,
                    "pokemon_score": 1.0,
                    "top_labels": []
                }
        else:
            precheck = {
                "is_pokemon": True,
                "pokemon_score": 1.0,
                "top_labels": []
            }
        try:
            raw_predictions = pokemon_model.predict_topk(temp_path, k=3) if precheck["is_pokemon"] else []
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")
        predictions = []
        for pred in raw_predictions:
            key = normalize_key(pred["name"])
            dataset_entry = DATASET.get(key)
            predictions.append({
                "name": dataset_entry["name"] if dataset_entry else pred["name"],
                "confidence": pred["confidence"]
            })
        top_label = precheck["top_labels"][0]["label"] if precheck["top_labels"] else "unknown"
        if precheck_model:
            explanation = f"Precheck compares the image to vision-language prompts. Top match: {top_label}."
        else:
            explanation = f"Precheck unavailable: {precheck_error or 'model not loaded'}."
        return {
            "top_3_predictions": predictions,
            "precheck": precheck,
            "explanation": explanation
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/pokemon/search")
def search_pokemon(query: str = ""):
    q = normalize_key(query)
    if not q:
        return {"results": []}
    results = []
    for entry in DATASET.values():
        if normalize_key(entry["name"]).startswith(q):
            results.append(entry["name"])
        if len(results) >= 8:
            break
    return {"results": results}

@app.get("/pokemon/{name}")
def get_pokemon(name: str):
    return fetch_pokemon_data(name)

@app.get("/type-effectiveness/{type_name}")
def get_type_effectiveness(type_name: str):
    return fetch_type_effectiveness(type_name)
