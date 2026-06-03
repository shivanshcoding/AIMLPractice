import SearchBar from "@/components/SearchBar";

export default function Navbar({ onSelect }) {
  return (
    <nav className="flex w-full flex-col items-center gap-4 rounded-2xl border border-white/10 bg-slate-950/50 px-6 py-5 text-center backdrop-blur-xl lg:flex-row lg:justify-between lg:text-left">
      <div className="flex items-center gap-3">
        <div className="h-2.5 w-2.5 rounded-full bg-cyan-300 shadow-[0_0_14px_rgba(56,189,248,0.8)]" />
        <div>
          <p className="text-heading text-lg text-white">Pokémon Identifier</p>
          <p className="text-xs text-slate-500">Vision Pokédex</p>
        </div>
      </div>
      <SearchBar onSelect={onSelect} />
    </nav>
  );
}
