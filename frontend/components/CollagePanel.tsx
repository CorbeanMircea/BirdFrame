interface Props {
  collageKey: number;
  generating: boolean;
  onGenerate: () => void;
  apiBase: string;
}

export default function CollagePanel({
  collageKey,
  generating,
  onGenerate,
  apiBase,
}: Props) {
  // collageKey=0 means not yet mounted — don't render image src yet
  const imageSrc =
    collageKey > 0
      ? `${apiBase}/collage/latest?t=${collageKey}`
      : null;

  return (
    <div className="bg-white rounded-lg border border-stone-200 overflow-hidden shadow-sm">
      <div className="flex items-center justify-between px-5 py-3 border-b border-stone-100">
        <h2 className="font-semibold text-stone-700">Heard Recently — Collage</h2>
        <button
          onClick={onGenerate}
          disabled={generating}
          className={`text-sm px-4 py-1.5 rounded transition-colors ${
            generating
              ? "bg-stone-200 text-stone-400 cursor-not-allowed"
              : "bg-stone-700 hover:bg-stone-600 text-white"
          }`}
        >
          {generating ? "Generating…" : "⟳ Generate"}
        </button>
      </div>

      <div className="relative bg-stone-100" style={{ aspectRatio: "16/9" }}>
        {imageSrc && (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            key={collageKey}
            src={imageSrc}
            alt="Heard Recently collage"
            className="w-full h-full object-contain"
            onError={(e) => {
              (e.target as HTMLImageElement).style.display = "none";
            }}
          />
        )}
        {generating && (
          <div className="absolute inset-0 bg-white/70 flex items-center justify-center">
            <div className="text-stone-500 text-sm animate-pulse">
              Generating collage…
            </div>
          </div>
        )}
        {!imageSrc && !generating && (
          <div className="absolute inset-0 flex items-center justify-center text-stone-400 text-sm">
            Click Generate to create a collage.
          </div>
        )}
      </div>

      <div className="px-5 py-2 text-xs text-stone-400 border-t border-stone-100">
        Click Generate to compose a new collage from recently detected species.
      </div>
    </div>
  );
}