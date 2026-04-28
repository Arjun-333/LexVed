"use client";

interface CitationCardProps {
  citations: string[];
}

function parseCitation(raw: string): { file: string; page: string | null } {
  // Parse "filename.pdf, Page: 4" format
  const parts = raw.split(",").map(s => s.trim());
  const file = parts[0] || raw;
  const pageMatch = raw.match(/Page:\s*(\d+)/i);
  return { file, page: pageMatch ? pageMatch[1] : null };
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
        {citations.map((c, i) => {
          const { file, page } = parseCitation(c);
          return (
            <button
              key={i}
              onClick={() => {
                // Highlight citation — could open a viewer in the future
                navigator.clipboard.writeText(c);
              }}
              className="group flex items-center gap-1 px-2 py-0.5 rounded font-mono text-[0.72rem] transition-all duration-300 cursor-pointer hover:scale-105"
              style={{
                background: "var(--surface-hover)",
                border: "1px solid var(--border)",
                color: "var(--text-secondary)",
              }}
              title={`Click to copy: ${c}`}
            >
              <span className="material-icons-round text-[11px] opacity-40 group-hover:opacity-100 group-hover:text-[var(--accent)] transition-all">
                description
              </span>
              <span>{file}</span>
              {page && (
                <span className="text-[0.6rem] px-1 py-0 rounded bg-[var(--accent-bg)] text-[var(--accent)] border border-[var(--accent-glow)] ml-0.5">
                  p.{page}
                </span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
