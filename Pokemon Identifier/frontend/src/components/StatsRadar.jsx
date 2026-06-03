import { motion } from "framer-motion";

const labels = ["hp", "attack", "defense", "spatk", "spdef", "speed"];

function polarPoint(angle, radius) {
  return {
    x: 50 + radius * Math.cos(angle),
    y: 50 + radius * Math.sin(angle)
  };
}

export default function StatsRadar({ stats }) {
  const values = labels.map((label) => Math.min(1, stats[label] / 150));
  const points = values
    .map((value, index) => {
      const angle = (Math.PI * 2 * index) / labels.length - Math.PI / 2;
      const point = polarPoint(angle, value * 34);
      return `${point.x},${point.y}`;
    })
    .join(" ");

  return (
    <div className="relative flex items-center justify-center">
      <svg viewBox="0 0 100 100" className="h-40 w-40">
        {[0.25, 0.5, 0.75, 1].map((scale) => (
          <polygon
            key={scale}
            points={labels
              .map((_, index) => {
                const angle = (Math.PI * 2 * index) / labels.length - Math.PI / 2;
                const point = polarPoint(angle, scale * 34);
                return `${point.x},${point.y}`;
              })
              .join(" ")}
            fill="none"
            stroke="rgba(148,163,255,0.12)"
            strokeWidth="0.6"
          />
        ))}
        <motion.polygon
          points={points}
          fill="rgba(56,189,248,0.25)"
          stroke="rgba(56,189,248,0.6)"
          strokeWidth="1.2"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6 }}
        />
      </svg>
      <div className="absolute grid grid-cols-2 gap-x-6 gap-y-2 text-[10px] uppercase tracking-[0.2em] text-slate-300">
        {labels.map((label) => (
          <span key={label}>{label}</span>
        ))}
      </div>
    </div>
  );
}
