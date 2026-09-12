"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

const API = "http://127.0.0.1:8000/api";

interface Species {
  id: number;
  scientific_name: string;
  common_name: string;
  artwork_path: string | null;
  detection_count: number;
}

export default function SpeciesPage() {
  const [species, setSpecies] = useState<Species[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");

  useEffect(() => {
    fetch(`${API}/species`)
      .then((r) => r.json())
      .then((data) => { setSpecies(data); setLoading(false); })
      .catch(() => { setError("Could not reach the API."); setLoading(false); });
  }, []);

  const filtered = species.filter(
    (s) =>
      s.common_name.toLowerCase().includes(search.toLowerCase()) ||
      s.scientific_name.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <main className="min-h-screen bg-stone-50 text-stone-800">
      <header className="bg-stone-800 text-stone-100 px-6 py-4 flex items-center justify-between shadow-md">
        <div className="flex items-center gap-3">
          <span className="text-2xl">🦜</span>
          <div>
            <h1 className="text-xl font-bold tracking-wide">BirdFrame</h1>
            <p className="text-stone-400 text-xs">Bird Sound Detection System</p>
          </div>
        </div>
        <nav className="flex gap-4 text-sm">
          <Link href="/" className="text-stone-300 hover:text-white transition-colors">Dashboard</Link>
          <Link href="/species" className="text-white font-medium border-b border-stone-400 pb-0.5">Species</Link>
          <Link href="/history" className="text-stone-300 hover:text-white transition-colors">History</Link>
        </nav>
      </header>

      <div className="max-w-screen-xl mx-auto px-6 py-6">
        <div className="flex items-center justify-between mb-6 gap-4 flex-wrap">
          <div>
            <h2 className="text-2xl font-bold text-stone-800">Detected Species</h2>
            <p className="text-stone-500 text-sm mt-0.5">
              {species.length} species identified so far
            </p>
          </div>
          <input
            type="text"
            placeholder="Search species…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="border border-stone-300 rounded-lg px-4 py-2 text-sm w-64
                       focus:outline-none focus:ring-2 focus:ring-stone-400 bg-white"
          />
        </div>

        {loading && <div className="text-center py-20 text-stone-400">Loading species…</div>}
        {error && <div className="text-center py-20 text-red-500">{error}</div>}
        {!loading && !error && filtered.length === 0 && (
          <div className="text-center py-20 text-stone-400">
            {search ? `No species match "${search}".` : "No species detected yet."}
          </div>
        )}

        {!loading && !error && filtered.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
            {filtered.map((s) => <SpeciesCard key={s.id} species={s} />)}
          </div>
        )}
      </div>
    </main>
  );
}

function SpeciesCard({ species }: { species: Species }) {
  return (
    <div className="bg-white rounded-lg border border-stone-200 shadow-sm overflow-hidden
                    hover:shadow-md hover:border-stone-300 transition-all">
      <div className="bg-stone-50 h-48 flex items-center justify-center overflow-hidden">
        {species.artwork_path ? (
          <ArtworkImage species={species} />
        ) : (
          <div className="text-stone-300 text-center px-4">
            <div className="text-4xl mb-2">🐦</div>
            <div className="text-xs">Generating…</div>
          </div>
        )}
      </div>
      <div className="px-4 py-3">
        <h3 className="font-semibold text-stone-800 text-sm leading-tight">
          {species.common_name}
        </h3>
        <p className="text-stone-400 text-xs italic mt-0.5 mb-2">
          {species.scientific_name}
        </p>
        <div className="flex items-center justify-between">
          <span className="text-xs text-stone-500">
            {species.detection_count} detection{species.detection_count !== 1 ? "s" : ""}
          </span>
          {species.artwork_path && (
            <span className="text-xs text-green-600 bg-green-50 px-2 py-0.5 rounded-full">
              {species.artwork_path.startsWith("generated/") ? "✨ Generated" : "✓ Artwork"}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

function ArtworkImage({ species }: { species: Species }) {
  const [failed, setFailed] = useState(false);

  // artwork_path is either "generated/stem" or "stem"
  // API endpoint is /api/species-artwork/<artwork_path>
  const src = `${API}/species-artwork/${species.artwork_path}`;

  if (failed) {
    return (
      <div className="text-stone-300 text-center px-4">
        <div className="text-4xl mb-2">🐦</div>
        <div className="text-xs">No illustration</div>
      </div>
    );
  }

  // Generated PNGs have transparent backgrounds — show on white
  const isGenerated = species.artwork_path?.startsWith("generated/");

  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={src}
      alt={species.common_name}
      className={`h-full object-contain ${isGenerated ? "p-4" : "w-full p-2"}`}
      onError={() => setFailed(true)}
    />
  );
}