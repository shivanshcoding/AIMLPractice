"use client";

import { useCallback, useState } from "react";
import { motion } from "framer-motion";

export default function UploadZone({ onFileSelected }) {
  const [dragging, setDragging] = useState(false);

  const handleDrop = useCallback(
    (event) => {
      event.preventDefault();
      setDragging(false);
      const file = event.dataTransfer.files?.[0];
      if (file) {
        onFileSelected(file);
      }
    },
    [onFileSelected]
  );

  return (
    <motion.div
      onDragOver={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      className={`glass group flex w-full flex-col items-center justify-center gap-4 rounded-3xl border border-dashed px-6 py-10 text-center transition ${
        dragging ? "border-cyan-400/70 bg-cyan-400/10" : "border-white/10"
      }`}
      whileHover={{ scale: 1.01 }}
    >
      <div className="flex h-12 w-12 items-center justify-center rounded-full border border-cyan-300/40 bg-cyan-300/10 text-cyan-200 shadow-[0_0_18px_rgba(56,189,248,0.4)]">
        ⭑
      </div>
      <div>
        <p className="text-heading text-lg text-white">Drop Pokémon image</p>
        <p className="text-sm text-slate-400">or click to upload from your device</p>
      </div>
      <label className="cursor-pointer rounded-full bg-cyan-500/20 px-4 py-2 text-xs uppercase tracking-[0.3em] text-cyan-200 transition hover:bg-cyan-500/40">
        Browse Files
        <input
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) {
              onFileSelected(file);
            }
          }}
        />
      </label>
    </motion.div>
  );
}
