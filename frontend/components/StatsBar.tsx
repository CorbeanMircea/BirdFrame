import { Stats } from "@/app/page";

interface Props {
  stats: Stats | null;
}

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-white rounded-lg border border-stone-200 px-5 py-4 flex-1 min-w-0">
      <p className="text-stone-500 text-xs uppercase tracking-widest mb-1">{label}</p>
      <p className="text-2xl font-bold text-stone-800">{value}</p>
    </div>
  );
}

export default function StatsBar({ stats }: Props) {
  const formatDate = (iso: string | null) => {
    if (!iso) return "—";
    return new Date(iso).toLocaleString();
  };

  return (
    <div className="flex gap-4 flex-wrap">
      <StatCard label="Total Detections" value={stats?.total_detections ?? "—"} />
      <StatCard label="Species Identified" value={stats?.total_species ?? "—"} />
      <StatCard label="Detection Events" value={stats?.total_events ?? "—"} />
      <StatCard
        label="Last Detection"
        value={formatDate(stats?.latest_detection_at ?? null)}
      />
    </div>
  );
}