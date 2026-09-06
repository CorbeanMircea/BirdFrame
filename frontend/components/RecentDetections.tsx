import { Detection } from "@/app/page";

interface Props {
  detections: Detection[];
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

export default function RecentDetections({ detections }: Props) {
  const formatTime = (iso: string) => {
    const d = new Date(iso);
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  };

  return (
    <div className="bg-white rounded-lg border border-stone-200 shadow-sm">
      <div className="px-5 py-3 border-b border-stone-100">
        <h2 className="font-semibold text-stone-700">Recent Detections</h2>
      </div>

      {detections.length === 0 ? (
        <div className="px-5 py-8 text-center text-stone-400 text-sm">
          No detections yet. Start the audio pipeline to begin.
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-stone-400 uppercase tracking-wider border-b border-stone-100">
                <th className="px-5 py-2 font-medium">Time</th>
                <th className="px-5 py-2 font-medium">Common Name</th>
                <th className="px-5 py-2 font-medium">Scientific Name</th>
                <th className="px-5 py-2 font-medium">Confidence</th>
                <th className="px-5 py-2 font-medium">Model</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-stone-50">
              {detections.map((d) => (
                <tr key={d.id} className="hover:bg-stone-50 transition-colors">
                  <td className="px-5 py-2.5 text-stone-500 font-mono text-xs">
                    {formatTime(d.timestamp)}
                  </td>
                  <td className="px-5 py-2.5 font-medium text-stone-800">
                    {d.common_name}
                  </td>
                  <td className="px-5 py-2.5 text-stone-500 italic">
                    {d.scientific_name}
                  </td>
                  <td className="px-5 py-2.5">
                    <ConfidenceBadge value={d.confidence} />
                  </td>
                  <td className="px-5 py-2.5 text-stone-400 text-xs">
                    {d.model_name ?? "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}