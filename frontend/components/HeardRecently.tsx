import { Species } from "@/app/page";

interface Props {
  species: Species[];
}

export default function HeardRecently({ species }: Props) {
  return (
    <div className="bg-white rounded-lg border border-stone-200 shadow-sm h-full">
      <div className="px-5 py-3 border-b border-stone-100">
        <h2 className="font-semibold text-stone-700">Species Heard Recently</h2>
      </div>

      {species.length === 0 ? (
        <div className="px-5 py-8 text-center text-stone-400 text-sm">
          No species detected yet.
          <br />
          <span className="text-xs">Start the audio pipeline to begin detection.</span>
        </div>
      ) : (
        <ul className="divide-y divide-stone-50">
          {species.map((s) => (
            <li key={s.id} className="px-5 py-3 flex items-center justify-between gap-3">
              <div className="min-w-0">
                <p className="font-medium text-stone-800 text-sm truncate">
                  {s.common_name}
                </p>
                <p className="text-stone-400 text-xs italic truncate">
                  {s.scientific_name}
                </p>
              </div>
              <span className="text-xs text-stone-500 bg-stone-100 rounded-full px-2 py-0.5 whitespace-nowrap">
                {s.detection_count} detection{s.detection_count !== 1 ? "s" : ""}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}