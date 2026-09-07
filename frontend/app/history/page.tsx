"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";

const API = "http://127.0.0.1:8000/api";

interface Detection {
  id: number;
  scientific_name: string;
  common_name: string;
  confidence: number;
  timestamp: string;
  duration_seconds: number | null;
  model_name: string | null;
  model_version: string | null;
  grouped_event_id: number | null;
}

interface DetectionList {
  total: number;
  limit: number;
  offset: number;
  items: Detection[];
}

interface Stats {
  total_detections: number;
  total_species: number;
  total_events: number;
  latest_detection_at: string | null;
}

const PAGE_SIZE = 25;

export default function HistoryPage() {
  const [detections, setDetections] = useState<Detection[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filterSpecies, setFilterSpecies] = useState("");
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const fetchDetections = useCallback(async (currentOffset: number) => {
    setLoading(true);
    try {
      const [detectRes, statsRes] = await Promise.all([
        fetch(`${API}/detections?limit=${PAGE_SIZE}&offset=${currentOffset}`),
        fetch(`${API}/stats`),
      ]);

      if (!detectRes.ok || !statsRes.ok) {
        setError("Could not reach the API. Is the server running?");
        return;
      }

      const detectData: DetectionList = await detectRes.json();
      const statsData: Stats = await statsRes.json();

      setDetections(detectData.items);
      setTotal(detectData.total);
      setStats(statsData);
      setError(null);
    } catch {
      setError("Could not reach the API. Is the server running?");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDetections(offset);
  }, [fetchDetections, offset]);

  const totalPages = Math.ceil(total / PAGE_SIZE);
  const currentPage = Math.floor(offset / PAGE_SIZE) + 1;

  const filtered = filterSpecies
    ? detections.filter(
        (d) =>
          d.common_name.toLowerCase().includes(filterSpecies.toLowerCase()) ||
          d.scientific_name.toLowerCase().includes(filterSpecies.toLowerCase())
      )
    : detections;

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
        <nav className="flex gap-4 text-sm">
          <Link href="/" className="text-stone-300 hover:text-white transition-colors">
            Dashboard
          </Link>
          <Link href="/species" className="text-stone-300 hover:text-white transition-colors">
            Species
          </Link>
          <Link
            href="/history"
            className="text-white font-medium border-b border-stone-400 pb-0.5"
          >
            History
          </Link>
        </nav>
      </header>

      <div className="max-w-screen-xl mx-auto px-6 py-6 space-y-6">
        {/* Page title + summary stats */}
        <div className="flex items-start justify-between gap-4 flex-wrap">
          <div>
            <h2 className="text-2xl font-bold text-stone-800">Detection History</h2>
            <p className="text-stone-500 text-sm mt-0.5">
              {total} total detection{total !== 1 ? "s" : ""} across{" "}
              {stats?.total_species ?? "—"} species
            </p>
          </div>
          <input
            type="text"
            placeholder="Filter by species…"
            value={filterSpecies}
            onChange={(e) => setFilterSpecies(e.target.value)}
            className="border border-stone-300 rounded-lg px-4 py-2 text-sm w-64
                       focus:outline-none focus:ring-2 focus:ring-stone-400 bg-white"
          />
        </div>

        {/* Summary stat cards */}
        {stats && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <SummaryCard label="Total Detections" value={stats.total_detections} />
            <SummaryCard label="Species" value={stats.total_species} />
            <SummaryCard label="Events" value={stats.total_events} />
            <SummaryCard
              label="Latest Detection"
              value={
                mounted && stats.latest_detection_at
                  ? new Date(stats.latest_detection_at).toLocaleDateString()
                  : stats.latest_detection_at
                  ? "—"
                  : "Never"
              }
            />
          </div>
        )}

        {/* Timeline */}
        <div className="bg-white rounded-lg border border-stone-200 shadow-sm">
          <div className="px-5 py-3 border-b border-stone-100 flex items-center justify-between">
            <h3 className="font-semibold text-stone-700">
              Detections — Page {currentPage} of {totalPages || 1}
            </h3>
            <button
              onClick={() => fetchDetections(offset)}
              className="text-xs text-stone-500 hover:text-stone-700 transition-colors"
            >
              ↻ Refresh
            </button>
          </div>

          {loading && (
            <div className="py-16 text-center text-stone-400">
              Loading detections…
            </div>
          )}

          {error && (
            <div className="py-16 text-center text-red-500">{error}</div>
          )}

          {!loading && !error && filtered.length === 0 && (
            <div className="py-16 text-center text-stone-400">
              {filterSpecies
                ? `No detections match "${filterSpecies}".`
                : "No detections recorded yet. Start the audio pipeline to begin."}
            </div>
          )}

          {!loading && !error && filtered.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs text-stone-400 uppercase tracking-wider border-b border-stone-100">
                    <th className="px-5 py-3 font-medium">Date / Time</th>
                    <th className="px-5 py-3 font-medium">Common Name</th>
                    <th className="px-5 py-3 font-medium">Scientific Name</th>
                    <th className="px-5 py-3 font-medium">Confidence</th>
                    <th className="px-5 py-3 font-medium">Duration</th>
                    <th className="px-5 py-3 font-medium">Event</th>
                    <th className="px-5 py-3 font-medium">Model</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-stone-50">
                  {filtered.map((d) => (
                    <DetectionRow key={d.id} detection={d} mounted={mounted} />
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Pagination */}
          {!loading && totalPages > 1 && (
            <div className="px-5 py-4 border-t border-stone-100 flex items-center justify-between">
              <button
                onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
                disabled={offset === 0}
                className={`px-4 py-1.5 rounded text-sm transition-colors ${
                  offset === 0
                    ? "text-stone-300 cursor-not-allowed"
                    : "bg-stone-100 hover:bg-stone-200 text-stone-700"
                }`}
              >
                ← Previous
              </button>

              <span className="text-sm text-stone-500">
                Showing {offset + 1}–{Math.min(offset + PAGE_SIZE, total)} of {total}
              </span>

              <button
                onClick={() => setOffset(offset + PAGE_SIZE)}
                disabled={offset + PAGE_SIZE >= total}
                className={`px-4 py-1.5 rounded text-sm transition-colors ${
                  offset + PAGE_SIZE >= total
                    ? "text-stone-300 cursor-not-allowed"
                    : "bg-stone-100 hover:bg-stone-200 text-stone-700"
                }`}
              >
                Next →
              </button>
            </div>
          )}
        </div>

        {/* Timeline visualisation — daily detection counts */}
        <DailyTimeline detections={detections} mounted={mounted} />
      </div>
    </main>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function SummaryCard({
  label,
  value,
}: {
  label: string;
  value: string | number;
}) {
  return (
    <div className="bg-white rounded-lg border border-stone-200 px-5 py-4">
      <p className="text-stone-500 text-xs uppercase tracking-widest mb-1">
        {label}
      </p>
      <p className="text-2xl font-bold text-stone-800">{value}</p>
    </div>
  );
}

function ConfidenceBadge({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const colour =
    pct >= 80
      ? "bg-green-100 text-green-700"
      : pct >= 60
      ? "bg-yellow-100 text-yellow-700"
      : "bg-red-100 text-red-700";
  return (
    <span className={`text-xs font-mono px-2 py-0.5 rounded-full ${colour}`}>
      {pct}%
    </span>
  );
}

function DetectionRow({
  detection: d,
  mounted,
}: {
  detection: Detection;
  mounted: boolean;
}) {
  const dt = new Date(d.timestamp);

  return (
    <tr className="hover:bg-stone-50 transition-colors">
      <td className="px-5 py-2.5 text-stone-500 text-xs font-mono whitespace-nowrap">
        {mounted ? (
          <>
            <span className="text-stone-400">
              {dt.toLocaleDateString([], {
                month: "short",
                day: "numeric",
              })}
            </span>
            <span className="ml-2">
              {dt.toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit",
              })}
            </span>
          </>
        ) : (
          <span className="text-stone-300">—</span>
        )}
      </td>
      <td className="px-5 py-2.5 font-medium text-stone-800">
        {d.common_name}
      </td>
      <td className="px-5 py-2.5 text-stone-500 italic text-xs">
        {d.scientific_name}
      </td>
      <td className="px-5 py-2.5">
        <ConfidenceBadge value={d.confidence} />
      </td>
      <td className="px-5 py-2.5 text-stone-400 text-xs">
        {d.duration_seconds != null
          ? `${d.duration_seconds.toFixed(1)}s`
          : "—"}
      </td>
      <td className="px-5 py-2.5 text-stone-400 text-xs">
        {d.grouped_event_id != null ? (
          <span className="bg-stone-100 px-2 py-0.5 rounded-full">
            #{d.grouped_event_id}
          </span>
        ) : (
          "—"
        )}
      </td>
      <td className="px-5 py-2.5 text-stone-400 text-xs">
        {d.model_name ?? "—"}
        {d.model_version ? ` ${d.model_version}` : ""}
      </td>
    </tr>
  );
}

function DailyTimeline({
  detections,
  mounted,
}: {
  detections: Detection[];
  mounted: boolean;
}) {
  if (detections.length === 0) return null;

  // Count detections per day
  const dayCounts: Record<string, number> = {};
  for (const d of detections) {
    const day = d.timestamp.slice(0, 10); // YYYY-MM-DD
    dayCounts[day] = (dayCounts[day] ?? 0) + 1;
  }

  const days = Object.entries(dayCounts).sort(([a], [b]) => a.localeCompare(b));
  const maxCount = Math.max(...days.map(([, c]) => c));

  return (
    <div className="bg-white rounded-lg border border-stone-200 shadow-sm px-5 py-4">
      <h3 className="font-semibold text-stone-700 mb-4">
        Daily Detection Activity
      </h3>
      <div className="flex items-end gap-2 h-24">
        {days.map(([day, count]) => (
          <div key={day} className="flex flex-col items-center flex-1 min-w-0">
            <div
              className="w-full bg-stone-300 rounded-t-sm transition-all"
              style={{ height: `${Math.round((count / maxCount) * 80)}px` }}
              title={`${count} detection${count !== 1 ? "s" : ""}`}
            />
            {mounted && (
              <span className="text-stone-400 text-xs mt-1 truncate w-full text-center">
                {new Date(day + "T12:00:00").toLocaleDateString([], {
                  month: "short",
                  day: "numeric",
                })}
              </span>
            )}
          </div>
        ))}
      </div>
      <p className="text-xs text-stone-400 mt-2">
        Showing activity from current page ({detections.length} detections)
      </p>
    </div>
  );
}