"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { searchPokemonNames } from "@/lib/api";

export default function SearchBar({ onSelect }) {
  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [open, setOpen] = useState(false);
  const timerRef = useRef(null);

  const debouncedFetch = useCallback((value) => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
    }
    timerRef.current = setTimeout(async () => {
      const results = await searchPokemonNames(value);
      setSuggestions(results.map((name) => ({ name })));
      setOpen(true);
    }, 250);
  }, []);

  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
    };
  }, []);

  return (
    <div className="relative w-full max-w-sm">
      <input
        value={query}
        onChange={(event) => {
          const value = event.target.value;
          setQuery(value);
          if (!value.trim()) {
            setSuggestions([]);
            setOpen(false);
            return;
          }
          debouncedFetch(value);
        }}
        placeholder="Search Pokémon..."
        className="w-full rounded-2xl border border-white/10 bg-slate-950/70 px-4 py-2 text-sm text-slate-100 shadow-[0_0_12px_rgba(15,23,42,0.45)] outline-none transition focus:border-cyan-300/60 focus:ring-2 focus:ring-cyan-400/20"
      />
      <AnimatePresence>
        {open ? (
          <motion.div
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 6 }}
            className="absolute left-0 right-0 z-20 mt-2 rounded-2xl border border-white/10 bg-slate-950/90 p-2 shadow-[0_0_18px_rgba(56,189,248,0.15)]"
          >
            {suggestions.length > 0 ? (
              suggestions.map((item) => (
                <button
                  key={item.name}
                  onClick={() => {
                    onSelect(item.name);
                    setQuery("");
                    setOpen(false);
                  }}
                  className="flex w-full items-center justify-between rounded-xl px-3 py-2 text-left text-sm text-slate-200 transition hover:bg-slate-800/70"
                >
                  <span>{item.name}</span>
                  <span className="text-xs text-cyan-300">Open</span>
                </button>
              ))
            ) : (
              <div className="rounded-xl px-3 py-2 text-xs text-slate-500">No results</div>
            )}
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  );
}
