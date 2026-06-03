import { motion } from "framer-motion";

export default function Loader({ label = "Scanning Pokédex..." }) {
  return (
    <div className="flex flex-col items-center gap-4">
      <motion.div
        className="relative h-16 w-16 rounded-full border-4 border-slate-500 bg-gradient-to-b from-red-500 via-red-500 to-slate-900 shadow-[0_0_24px_rgba(248,113,113,0.5)]"
        animate={{ rotate: 360 }}
        transition={{ repeat: Infinity, duration: 1.4, ease: "linear" }}
      >
        <div className="absolute inset-x-0 top-1/2 h-1 -translate-y-1/2 bg-slate-100/90" />
        <div className="absolute left-1/2 top-1/2 h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-4 border-slate-200 bg-slate-900" />
      </motion.div>
      <p className="text-xs uppercase tracking-[0.3em] text-slate-300">{label}</p>
    </div>
  );
}
