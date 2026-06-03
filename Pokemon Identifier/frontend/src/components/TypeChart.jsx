import { motion } from "framer-motion";

const tint = {
  strong: "bg-emerald-500/20 text-emerald-200 border-emerald-400/40",
  weak: "bg-rose-500/20 text-rose-200 border-rose-400/40",
  immune: "bg-slate-700/40 text-slate-200 border-slate-500/50",
  neutral: "bg-slate-800/50 text-slate-300 border-slate-700/60"
};

export default function TypeChart({ effectiveness }) {
  const strong = effectiveness.double_damage_to;
  const weak = effectiveness.half_damage_to;
  const immune = effectiveness.no_damage_to;

  const renderGroup = (title, items, style) => (
    <div className="flex flex-col gap-2">
      <p className="text-xs uppercase tracking-[0.3em] text-slate-400">{title}</p>
      <div className="flex flex-wrap gap-2">
        {items.length === 0 ? (
          <span className={`rounded-full border px-3 py-1 text-xs ${tint.neutral}`}>None</span>
        ) : (
          items.map((item) => (
            <span key={item} className={`rounded-full border px-3 py-1 text-xs ${style}`}>
              {item}
            </span>
          ))
        )}
      </div>
    </div>
  );

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="grid gap-4"
    >
      {renderGroup("Strong Against", strong, tint.strong)}
      {renderGroup("Weak Against", weak, tint.weak)}
      {renderGroup("No Effect", immune, tint.immune)}
    </motion.div>
  );
}
