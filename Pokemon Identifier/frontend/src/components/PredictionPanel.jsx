import { motion } from "framer-motion";

export default function PredictionPanel({ predictions, active, onSelect }) {
  return (
    <div className="glass flex flex-col gap-4 rounded-2xl p-5">
      <div className="flex items-center justify-between">
        <p className="text-heading text-xs uppercase tracking-[0.3em] text-cyan-200">Top Predictions</p>
        <span className="text-xs text-slate-500">Tap to inspect</span>
      </div>
      <div className="grid gap-3">
        {predictions.map((prediction, index) => (
          <motion.button
            key={prediction.name}
            onClick={() => onSelect(prediction.name)}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className={`group flex flex-col gap-2 rounded-2xl border px-4 py-3 text-left transition ${
              active === prediction.name
                ? "border-cyan-400/70 bg-cyan-500/10 shadow-[0_0_14px_rgba(56,189,248,0.2)]"
                : "border-white/10 bg-slate-950/50 hover:border-cyan-400/40"
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="text-base text-white">{prediction.name}</span>
              <span className="text-xs text-cyan-200">
                {(prediction.confidence * 100).toFixed(1)}%
              </span>
            </div>
            <div className="h-2 w-full rounded-full bg-slate-800/70">
              <motion.div
                className="h-2 rounded-full bg-gradient-to-r from-cyan-400 to-indigo-400"
                initial={{ width: 0 }}
                animate={{ width: `${Math.min(100, prediction.confidence * 100)}%` }}
                transition={{ duration: 0.6 }}
              />
            </div>
          </motion.button>
        ))}
      </div>
    </div>
  );
}
