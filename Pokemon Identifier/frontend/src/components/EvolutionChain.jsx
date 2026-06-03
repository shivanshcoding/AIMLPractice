import { motion } from "framer-motion";

export default function EvolutionChain({ chain, onSelect }) {
  if (!chain.length) {
    return <p className="text-sm text-slate-400">No evolution data</p>;
  }

  return (
    <div className="flex flex-wrap items-center gap-3">
      {chain.map((name, index) => (
        <div key={name} className="flex items-center gap-3">
          <motion.button
            whileHover={{ scale: 1.05 }}
            onClick={() => onSelect(name)}
            className="rounded-full border border-cyan-400/30 bg-cyan-500/10 px-3 py-1 text-xs uppercase tracking-[0.2em] text-cyan-200"
          >
            {name}
          </motion.button>
          {index < chain.length - 1 ? (
            <motion.span
              initial={{ opacity: 0, x: -6 }}
              animate={{ opacity: 1, x: 0 }}
              className="text-cyan-400"
            >
              →
            </motion.span>
          ) : null}
        </div>
      ))}
    </div>
  );
}
