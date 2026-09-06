"use client";

import { useEffect, useState, useCallback } from "react";
import CollagePanel from "@/components/CollagePanel";
import StatsBar from "@/components/StatsBar";
import RecentDetections from "@/components/RecentDetections";
import HeardRecently from "@/components/HeardRecently";

const API = "http://127.0.0.1:8000/api";

export interface Stats {
  total_detections: number;
  total_species: number;
  total_events: number;
  latest_detection_at: string | null;
}

export interface Detection {
  id: number;
  scientific_name: string;
  common_name: string;
  confidence: number;
  timestamp: string;
  model_name: string | null;
}

export interface Species {
  id: number;
  scientific_name: string;
  common_name: string;
  artwork_path: string | null;
  detection_count: number;
}

export default function Dashboard() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [heardRecently, setHeardRecently] = useState<Species[]>([]);
  // Stable initial value — updated client-side only to avoid hydration mismatch
  const [collageKey, setCollageKey] = useState(0);
  const [generating, setGenerating] = useState(false);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [mounted, setMounted] = useState(false);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);

  useEffect(() => {
    setMounted(true);
    // Set stable collage key after mount
    setCollageKey(Date.now());
  }, []);

  const fetchAll = useCallback(async () => {
    try {
      const [statsRes, detectionsRes, heardRes] = await Promise.all([
        fetch(`${API}/stats`),
        fetch(`${API}/detections?limit=10`),
        fetch(`${API}/heard-recently?limit=6`),
      ]);

      if (!statsRes.ok || !detectionsRes.ok || !heardRes.ok) {
        setApiOnline(false);
        return;
      }

      const statsData = await statsRes.json();
      const detectionsData = await detectionsRes.json();
      const heardData = await heardRes.json();

      setStats(statsData);
      setDetections(detectionsData.items);
      setHeardRecently(heardData.species);
      setLastRefresh(new Date());
      setApiOnline(true);
    } catch {
      setApiOnline(false);
    }
  }, []);

  const generateCollage = async () => {
    setGenerating(true);
    try {
      await fetch(`${API}/collage/generate`, { method: "POST" });
      setCollageKey(Date.now());
    } finally {
      setGenerating(false);
    }
  };

  useEffect(() => {
    fetchAll();
    const interval = setInterval(fetchAll, 15_000);
    return () => clearInterval(interval);
  }, [fetchAll]);

  return (
    <main className="min-h-screen bg-stone-50 text-stone-800">
      {/* Header */}
      <header className="bg-stone-800 text-stone-100 px-6 py-4 flex items-center justify-between shadow-md">
        <div className="flex items-center gap-3">
          <span className="text-2xl">🦜</span>
          <div>
            <h1 className="text-xl font-bold tracking-wide">BirdFrame</h1>
            <p className="text-stone-400 text-xs">Bird Sound Detection System</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 text-sm">
            <span
              className={`w-2 h-2 rounded-full ${
                apiOnline === null
                  ? "bg-stone-400"
                  : apiOnline
                  ? "bg-green-400"
                  : "bg-red-400"
              }`}
            />
            <span className="text-stone-300">
              {apiOnline === null
                ? "Connecting…"
                : apiOnline
                ? "API online"
                : "API offline"}
            </span>
          </div>

          {mounted && lastRefresh && (
            <span className="text-stone-500 text-xs">
              Refreshed {lastRefresh.toLocaleTimeString()}
            </span>
          )}

          <button
            onClick={fetchAll}
            className="bg-stone-700 hover:bg-stone-600 px-3 py-1.5 rounded text-sm transition-colors"
          >
            ↻ Refresh
          </button>
        </div>
      </header>

      <div className="max-w-screen-xl mx-auto px-6 py-6 space-y-6">
        <StatsBar stats={stats} />

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <CollagePanel
              collageKey={collageKey}
              generating={generating}
              onGenerate={generateCollage}
              apiBase={API}
            />
          </div>
          <div>
            <HeardRecently species={heardRecently} />
          </div>
        </div>

        <RecentDetections detections={detections} />
      </div>
    </main>
  );
}