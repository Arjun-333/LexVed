"use client";

interface CitationCardProps {
  citations: string[];
}

export default function CitationCard({ citations }: CitationCardProps) {
  if (!citations.length) return null;

  return (
    <div
      className="mt-3 py-2.5 px-3 rounded-lg text-[0.78rem] transition-colors duration-300"
      style={{
        background: "var(--accent-subtle)",
        borderLeft: "2px solid var(--accent-dim)",
      }}
    >
      <div
        className="flex items-center gap-1.5 mb-2 text-[0.7rem] font-semibold uppercase tracking-[0.08em]"
        style={{ color: "var(--accent)" }}
      >
        <span className="material-icons-round text-[13px]">menu_book</span>
        Case References
      </div>
      <div className="flex flex-wrap gap-1.5">
        {citations.map((c, i) => (
          <span
            key={i}
            className="px-2 py-0.5 rounded font-mono text-[0.72rem] transition-colors duration-300"
            style={{
              background: "var(--surface-hover)",
              border: "1px solid var(--border)",
              color: "var(--text-secondary)",
            }}
          >
            {c}
          </span>
        ))}
      </div>
    </div>
  );
}
