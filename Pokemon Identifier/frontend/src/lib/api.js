const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export async function predictPokemon(file) {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`${API_URL}/predict`, {
    method: "POST",
    body: form
  });
  if (!response.ok) {
    let detail = "Prediction failed";
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch {
      detail = "Prediction failed";
    }
    throw new Error(detail);
  }
  return response.json();
}

export async function getPokemon(name) {
  const response = await fetch(`${API_URL}/pokemon/${encodeURIComponent(name)}`);
  if (!response.ok) {
    throw new Error("Pokemon not found");
  }
  return response.json();
}

export async function getTypeEffectiveness(typeName) {
  const response = await fetch(`${API_URL}/type-effectiveness/${encodeURIComponent(typeName)}`);
  if (!response.ok) {
    throw new Error("Type effectiveness not found");
  }
  return response.json();
}

export async function searchPokemonNames(query) {
  if (!query.trim()) {
    return [];
  }
  const response = await fetch(`${API_URL}/pokemon/search?query=${encodeURIComponent(query)}`);
  if (!response.ok) {
    return [];
  }
  const data = await response.json();
  return data.results ?? [];
}
