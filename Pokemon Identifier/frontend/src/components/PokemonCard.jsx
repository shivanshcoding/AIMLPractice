"use client";

import { useState } from "react";
import Image from "next/image";
import dynamic from "next/dynamic";
import { motion } from "framer-motion";
import { typeColors } from "@/lib/typeColors";
import EvolutionChain from "@/components/EvolutionChain";

const TypeChart = dynamic(() => import("@/components/TypeChart"), { ssr: false });
const StatsRadar = dynamic(() => import("@/components/StatsRadar"), { ssr: false });

export default function PokemonCard({ pokemon, effectiveness, onSelectEvolution }) {
  const [flipped, setFlipped] = useState(false);
  const type1Style = typeColors[pokemon.type1] || typeColors.Normal;
  const type2Style = pokemon.type2 ? typeColors[pokemon.type2] || typeColors.Normal : null;
  const apiBase = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  const imageSrc = pokemon.image_url
    ? pokemon.image_url.startsWith("http")
      ? pokemon.image_url
      : `${apiBase}${pokemon.image_url}`
    : null;

  return (
    <div className="relative mx-auto w-full max-w-[420px] min-w-[320px]">
      <motion.button
        onClick={() => setFlipped((prev) => !prev)}
        className="absolute right-5 top-5 z-10 rounded-full border border-white/10 bg-slate-950/70 px-3 py-1 text-[11px] uppercase tracking-[0.2em] text-slate-200"
        whileHover={{ scale: 1.05 }}
      >
        {flipped ? "Front" : "Details"}
      </motion.button>
      <motion.div
        className="relative h-[520px] w-full rounded-[28px] [transform-style:preserve-3d]"
        animate={{ rotateY: flipped ? 180 : 0 }}
        transition={{ duration: 0.6 }}
      >
        <div className="glass absolute inset-0 flex h-full w-full flex-col justify-between rounded-[28px] p-6 [backface-visibility:hidden]">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.4em] text-slate-400">#{pokemon.num}</p>
              <h3 className="text-heading text-2xl text-white">{pokemon.name}</h3>
            </div>
            {pokemon.legendary ? (
              <span className="rounded-full border border-yellow-300/40 bg-yellow-400/20 px-3 py-1 text-[10px] uppercase tracking-[0.3em] text-yellow-100">
                Legendary
              </span>
            ) : null}
          </div>
          <div className="flex items-center justify-center">
            <div className="relative h-52 w-52 rounded-3xl bg-slate-950/40 p-3">
              {imageSrc ? (
                <Image
                  src={imageSrc}
                  alt={pokemon.name}
                  fill
                  className="object-contain drop-shadow-[0_0_24px_rgba(56,189,248,0.2)]"
                  sizes="224px"
                />
              ) : (
                <div className="flex h-full w-full items-center justify-center rounded-3xl border border-white/10 text-sm text-slate-400">
                  No Image
                </div>
              )}
            </div>
          </div>
          <div className="flex flex-col gap-4">
            <div className="flex flex-wrap gap-2">
              <span className={`rounded-full border px-3 py-1 text-xs ${type1Style}`}>
                {pokemon.type1}
              </span>
              {pokemon.type2 ? (
                <span className={`rounded-full border px-3 py-1 text-xs ${type2Style}`}>
                  {pokemon.type2}
                </span>
              ) : null}
            </div>
            <div className="grid grid-cols-3 gap-3 text-xs text-slate-300">
              <div className="rounded-xl border border-white/10 bg-slate-900/40 px-3 py-2">
                <p className="text-[10px] uppercase tracking-[0.2em] text-slate-500">HP</p>
                <p className="text-sm text-white">{pokemon.stats.hp}</p>
              </div>
              <div className="rounded-xl border border-white/10 bg-slate-900/40 px-3 py-2">
                <p className="text-[10px] uppercase tracking-[0.2em] text-slate-500">Attack</p>
                <p className="text-sm text-white">{pokemon.stats.attack}</p>
              </div>
              <div className="rounded-xl border border-white/10 bg-slate-900/40 px-3 py-2">
                <p className="text-[10px] uppercase tracking-[0.2em] text-slate-500">Defense</p>
                <p className="text-sm text-white">{pokemon.stats.defense}</p>
              </div>
            </div>
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span>Generation</span>
              <span className="text-slate-200">{pokemon.generation ?? "Unknown"}</span>
            </div>
          </div>
        </div>

        <div className="glass absolute inset-0 flex h-full w-full flex-col gap-6 rounded-[28px] p-6 [transform:rotateY(180deg)] [backface-visibility:hidden]">
          <div>
            <h4 className="text-heading text-xs uppercase tracking-[0.3em] text-cyan-200">Stats Radar</h4>
            <StatsRadar stats={pokemon.stats} />
          </div>
          <div className="grid gap-4">
            <h4 className="text-heading text-xs uppercase tracking-[0.3em] text-cyan-200">
              Type Effectiveness
            </h4>
            {effectiveness ? (
              <TypeChart effectiveness={effectiveness} />
            ) : (
              <p className="text-sm text-slate-400">No type data loaded</p>
            )}
          </div>
          <div className="grid gap-3">
            <h4 className="text-heading text-xs uppercase tracking-[0.3em] text-cyan-200">
              Evolution Chain
            </h4>
            <EvolutionChain chain={pokemon.evolution_chain} onSelect={onSelectEvolution} />
          </div>
        </div>
      </motion.div>
    </div>
  );
}
