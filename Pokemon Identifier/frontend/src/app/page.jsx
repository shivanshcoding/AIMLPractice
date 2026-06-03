"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Image from "next/image";
import { motion, AnimatePresence } from "framer-motion";
import Navbar from "@/components/Navbar";
import UploadZone from "@/components/UploadZone";
import PredictionPanel from "@/components/PredictionPanel";
import PokemonCard from "@/components/PokemonCard";
import Loader from "@/components/Loader";
import { getPokemon, getTypeEffectiveness, predictPokemon } from "@/lib/api";

export default function HomePage() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [precheck, setPrecheck] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [isRejected, setIsRejected] = useState(false);
  const [rejectConfidence, setRejectConfidence] = useState(null);
  const [loadingPrediction, setLoadingPrediction] = useState(false);
  const [selectedPokemon, setSelectedPokemon] = useState(null);
  const [pokemonData, setPokemonData] = useState(null);
  const [typeData, setTypeData] = useState(null);
  const [loadingDetails, setLoadingDetails] = useState(false);
  const [error, setError] = useState(null);

  const formatBytes = useCallback((bytes) => {
    if (!bytes) return "0 KB";
    const sizes = ["B", "KB", "MB"];
    const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), sizes.length - 1);
    return `${(bytes / Math.pow(1024, i)).toFixed(i === 0 ? 0 : 1)} ${sizes[i]}`;
  }, []);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [selectedFile]);

  const handlePredict = useCallback(async () => {
    if (!selectedFile) {
      setError("Upload an image to begin.");
      return;
    }
    setLoadingPrediction(true);
    setError(null);
    setIsRejected(false);
    setRejectConfidence(null);
    try {
      const response = await predictPokemon(selectedFile);
      setPredictions(response.top_3_predictions);
      setPrecheck(response.precheck);
      setExplanation(response.explanation);
      const top = response.top_3_predictions[0]?.name || null;
      setSelectedPokemon(top);
      if (!response.precheck?.is_pokemon) {
        setIsRejected(true);
        setRejectConfidence(response.precheck?.pokemon_score ?? null);
      }
    } catch (err) {
      setError(err?.message || "Prediction failed. Try another image.");
    } finally {
      setLoadingPrediction(false);
    }
  }, [selectedFile]);

  const fetchPokemonDetails = useCallback(async (name) => {
    setLoadingDetails(true);
    try {
      const data = await getPokemon(name);
      setPokemonData(data);
      if (data.type1) {
        const typeInfo = await getTypeEffectiveness(data.type1);
        setTypeData(typeInfo);
      } else {
        setTypeData(null);
      }
    } catch {
      setError("Pokédex data unavailable.");
    } finally {
      setLoadingDetails(false);
    }
  }, []);

  useEffect(() => {
    if (selectedPokemon && predictions.length > 0) {
      fetchPokemonDetails(selectedPokemon);
    }
  }, [selectedPokemon, predictions.length, fetchPokemonDetails]);

  const heroStats = useMemo(
    () => [
      { label: "Realtime Vision", value: "CLIP + Classifier" },
      { label: "Inference", value: "Top-3 Ranking" },
      { label: "Data", value: "Local Pokédex" }
    ],
    []
  );
  const hasPredictions = predictions.length > 0;

  return (
    <main className="relative min-h-screen px-4 pb-24 pt-10">
      <div className="absolute inset-0 overflow-hidden">
        <motion.div
          className="absolute left-1/2 top-20 h-72 w-72 -translate-x-1/2 rounded-full bg-purple-500/20 blur-[120px]"
          animate={{ opacity: [0.5, 0.8, 0.5], scale: [0.9, 1.1, 0.9] }}
          transition={{ duration: 6, repeat: Infinity }}
        />
        <motion.div
          className="absolute right-10 top-60 h-40 w-40 rounded-full bg-cyan-500/20 blur-[90px]"
          animate={{ opacity: [0.4, 0.7, 0.4], y: [0, -20, 0] }}
          transition={{ duration: 5, repeat: Infinity }}
        />
      </div>

      <div className="relative mx-auto flex w-full max-w-4xl flex-col gap-10">
        <Navbar onSelect={setSelectedPokemon} />

        <section className="grid gap-10 lg:grid-cols-[1fr_0.95fr] lg:items-start lg:gap-10">
          <div className="flex flex-col gap-6 text-center lg:text-left">
            <div className="glass-strong neon-border rounded-[32px] px-8 py-10">
              <div className="flex flex-col items-center gap-6 lg:flex-row lg:items-start lg:justify-between">
                <div className="space-y-4 lg:max-w-lg">
                  <p className="text-heading text-sm uppercase tracking-[0.4em] text-cyan-300">
                    Vision Pokédex
                  </p>
                  <h1 className="text-heading gradient-text text-4xl md:text-5xl">
                    Identify Pokémon in seconds.
                  </h1>
                  <p className="text-sm text-slate-300">
                    Upload an image, pass the vision-language gatekeeper, and reveal a premium Pokédex card.
                  </p>
                  <div className="flex flex-wrap justify-center gap-3 lg:justify-start">
                    {heroStats.map((stat) => (
                      <div
                        key={stat.label}
                        className="glass rounded-2xl border border-white/10 px-4 py-2 text-xs text-slate-300"
                      >
                        <p className="text-[10px] uppercase tracking-[0.3em] text-slate-500">
                          {stat.label}
                        </p>
                        <p className="text-sm text-white">{stat.value}</p>
                      </div>
                    ))}
                  </div>
                </div>

                <motion.div
                  className="relative flex h-36 w-36 items-center justify-center rounded-full border border-white/10 bg-slate-950/40"
                  animate={{ rotate: 360 }}
                  transition={{ repeat: Infinity, duration: 9, ease: "linear" }}
                >
                  <div className="absolute h-28 w-28 rounded-full border-4 border-red-500/80 bg-gradient-to-b from-red-500 to-slate-900 shadow-[0_0_30px_rgba(248,113,113,0.5)]" />
                  <div className="absolute inset-x-0 top-1/2 h-2 -translate-y-1/2 bg-slate-100/90" />
                  <div className="absolute h-8 w-8 rounded-full border-4 border-slate-200 bg-slate-900" />
                </motion.div>
              </div>
            </div>

            <UploadZone
              onFileSelected={(file) => {
                setSelectedFile(file);
                setPredictions([]);
                setPokemonData(null);
                setPrecheck(null);
                setExplanation(null);
              }}
            />

            {previewUrl ? (
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                className="glass-strong rounded-[28px] p-5"
              >
                <div className="flex flex-wrap items-center justify-between gap-4">
                  <div className="text-left">
                    <p className="text-heading text-xs uppercase tracking-[0.3em] text-cyan-300">
                      Upload Preview
                    </p>
                    <p className="mt-2 text-sm text-slate-200">{selectedFile?.name}</p>
                    <p className="text-xs text-slate-400">{formatBytes(selectedFile?.size || 0)}</p>
                  </div>
                  <div className="relative h-24 w-24 overflow-hidden rounded-2xl border border-white/10">
                    <Image
                      src={previewUrl}
                      alt="Preview"
                      fill
                      unoptimized
                      className="object-cover"
                      sizes="96px"
                    />
                  </div>
                </div>
              </motion.div>
            ) : null}

            <div className="flex flex-col items-center gap-4 sm:flex-row sm:justify-center lg:justify-start">
              <button
                onClick={handlePredict}
                disabled={!selectedFile || loadingPrediction}
                className="glow-button rounded-full bg-cyan-300/90 px-6 py-3 text-sm uppercase tracking-[0.3em] text-slate-900 transition hover:bg-cyan-200 disabled:cursor-not-allowed disabled:opacity-50"
              >
                Identify Pokémon
              </button>
              {previewUrl ? (
                <div className="flex items-center gap-3 text-xs text-slate-400">
                  <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_12px_rgba(52,211,153,0.7)]" />
                  {selectedFile?.name}
                </div>
              ) : null}
            </div>

            <AnimatePresence>
              {loadingPrediction ? (
                <motion.div
                  key="loader"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="glass rounded-3xl p-6"
                >
                  <Loader label="Analyzing image..." />
                </motion.div>
              ) : null}
            </AnimatePresence>

            {precheck ? (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="glass rounded-3xl p-6"
              >
                <div className="flex items-center justify-between">
                  <p className="text-heading text-xs uppercase tracking-[0.3em] text-cyan-300">
                    Image Check
                  </p>
                  <span
                    className={`rounded-full border px-3 py-1 text-xs ${
                      precheck.is_pokemon
                        ? "border-emerald-400/40 bg-emerald-400/20 text-emerald-200"
                        : "border-rose-400/40 bg-rose-400/20 text-rose-200"
                    }`}
                  >
                    {precheck.is_pokemon ? "Pokémon Likely" : "Not Pokémon"}
                  </span>
                </div>
                {precheck.is_pokemon ? (
                  <p className="mt-3 text-sm text-slate-300">Image looks like a Pokémon. Running full identification.</p>
                ) : (
                  <p className="mt-3 text-sm text-rose-200">
                    Uh oh — this image doesn’t look like a Pokémon.
                    {rejectConfidence !== null ? ` Confidence ${(rejectConfidence * 100).toFixed(1)}%.` : ""}
                  </p>
                )}
              </motion.div>
            ) : null}

            {error ? <p className="text-sm text-rose-300">{error}</p> : null}
          </div>

          <div className="flex flex-col items-center gap-6">
            {hasPredictions ? (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="w-full"
              >
                <PredictionPanel
                  predictions={predictions}
                  active={selectedPokemon ?? undefined}
                  onSelect={setSelectedPokemon}
                />
              </motion.div>
            ) : (
              <div className="glass flex w-full flex-col gap-3 rounded-2xl p-5 text-sm text-slate-400">
                <p className="text-heading text-xs uppercase tracking-[0.3em] text-cyan-200">Predictions</p>
                <p>Upload an image to unlock predictions and a premium Pokédex card.</p>
              </div>
            )}

            <div className="flex min-h-[520px] w-full items-center justify-center">
              {loadingDetails && hasPredictions ? (
                <div className="glass h-[520px] w-full max-w-[420px] animate-pulse rounded-[28px] bg-slate-900/60" />
              ) : hasPredictions && pokemonData && !isRejected ? (
                <motion.div
                  initial={{ opacity: 0, y: 30, scale: 0.98 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  transition={{ duration: 0.5, ease: "easeOut" }}
                >
                  <PokemonCard
                    pokemon={pokemonData}
                    effectiveness={typeData}
                    onSelectEvolution={setSelectedPokemon}
                  />
                </motion.div>
              ) : (
                <div className="flex h-[520px] w-full max-w-[420px] items-center justify-center text-sm text-slate-500">
                  No card yet. Run a prediction to reveal the Pokédex.
                </div>
              )}
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
