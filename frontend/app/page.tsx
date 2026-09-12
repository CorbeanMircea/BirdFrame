"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import Link from "next/link";
import CollagePanel from "@/components/CollagePanel";
import StatsBar from "@/components/StatsBar";
import RecentDetections from "@/components/RecentDetections";
import HeardRecently from "@/components/HeardRecently";

const API = "http://127.0.0.1:8000/api";
const POLL_INTERVAL = 15_000;       // fetch stats every 15s
const COLLAGE_AUTO_INTERVAL = 60_000; // auto-regenerate collage every 60s

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
  const [collageKey, setCollageKey] = useState(0);
  const [generating, setGenerating] = useState(false);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [mounted, setMounted] = useState(false);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);
  const [lastDetectionCount, setLastDetectionCount] = useState(0);
  const lastCollageGenRef = useRef<number>(0);

  useEffect(() => {
    setMounted(true);
    setCollageKey(Date.now());
  }, []);

  const generateCollage = useCallback(async (auto = false) => {
    if (generating) return;
    setGenerating(true);
    try {
      await fetch(`${API}/collage/generate`, { method: "POST" });
      setCollageKey(Date.now());
      lastCollageGenRef.current = Date.now();
      if (auto) {
        console.log("Auto-regenerated collage after new detections.");
      }
    } catch (e) {
      console.error("Collage generation failed:", e);
    } finally {
      setGenerating(false);
    }
  }, [generating]);

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

      const statsData: Stats = await statsRes.json();
      const detectionsData = await detectionsRes.json();
      const heardData = await heardRes.json();

      setStats(statsData);
      setDetections(detectionsData.items);
      setHeardRecently(heardData.species);
      setLastRefresh(new Date());
      setApiOnline(true);

      // Auto-regenerate collage if:
      // 1. New detections arrived since last check, AND
      // 2. At least 60s since last generation
      const newCount = statsData.total_detections;
      const timeSinceLastGen = Date.now() - lastCollageGenRef.current;
      if (
        newCount > lastDetectionCount &&
        lastDetectionCount > 0 &&
        timeSinceLastGen > COLLAGE_AUTO_INTERVAL
      ) {
        generateCollage(true);
      }
      setLastDetectionCount(newCount);

    } catch {
      setApiOnline(false);
    }
  }, [generateCollage, lastDetectionCount]);

  useEffect(() => {
    fetchAll();
    const interval = setInterval(fetchAll, POLL_INTERVAL);
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

        <div className="flex items-center gap-6">
          <nav className="flex gap-4 text-sm">
            <Link href="/"
              className="text-white font-medium border-b border-stone-400 pb-0.5">
              Dashboard
            </Link>
            <Link href="/species"
              className="text-stone-300 hover:text-white transition-colors">
              Species
            </Link>
            <Link href="/history"
              className="text-stone-300 hover:text-white transition-colors">
              History
            </Link>
          </nav>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-sm">
              <span className={`w-2 h-2 rounded-full ${
                apiOnline === null ? "bg-stone-400"
                : apiOnline ? "bg-green-400"
                : "bg-red-400"
              }`} />
              <span className="text-stone-300">
                {apiOnline === null ? "Connecting…"
                  : apiOnline ? "API online"
                  : "API offline"}
              </span>
            </div>

            {mounted && lastRefresh && (
              <span className="text-stone-500 text-xs">
                Refreshed {lastRefresh.toLocaleTimeString()}
              </span>
            )}

            <button
              onClick={() => fetchAll()}
              className="bg-stone-700 hover:bg-stone-600 px-3 py-1.5 rounded text-sm transition-colors"
            >
              ↻ Refresh
            </button>
          </div>
        </div>
      </header>

      <div className="max-w-screen-xl mx-auto px-6 py-6 space-y-6">
        <StatsBar stats={stats} />

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <CollagePanel
              collageKey={collageKey}
              generating={generating}
              onGenerate={() => generateCollage(false)}
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