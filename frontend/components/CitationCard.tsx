"use client";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:5000";

interface Source {
  file: string;
  page: number;
  path: string;
}

interface CitationCardProps {
  citations: string[];
  sources?: Source[];
  onCitationClick?: (file: string, page: number, text: string) => void;
}

function parseCitation(raw: string): { file: string; page: string | null } {
  const parts = raw.split(",").map(s => s.trim());
  const file = parts[0] || raw;
  const pageMatch = raw.match(/Page:\s*(\d+)/i);
  return { file, page: pageMatch ? pageMatch[1] : null };
}

function deduplicateSources(sources: Source[]): Source[] {
  const seen = new Set<string>();
  return sources.filter(s => {
    const key = `${s.file}|${s.page}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function openPdf(filename: string, page: number) {
  // Open PDF in a new tab via the backend API with page fragment and secure token
  const token = typeof window !== "undefined" ? localStorage.getItem("lexved_token") : null;
  
  if (token) {
    const finalFilename = filename.toLowerCase().endsWith(".pdf") ? filename : `${filename}.pdf`;
    // We use the 1-indexed page directly as PDF viewers expect #page=1 for the first page
    const finalUrl = `${API_URL}/api/pdf/${encodeURIComponent(finalFilename)}?token=${token}#page=${page + 1}`;
    
    // Open in a new tab
    window.open(finalUrl, "_blank");
  }
}

export default function CitationCard({ citations, sources, onCitationClick }: CitationCardProps) {
  // Merge parsed citations with structured sources
  const uniqueSources = sources ? deduplicateSources(sources) : [];
  
  // If we have structured sources, prefer those
  const hasSources = uniqueSources.length > 0;
  const displayItems = hasSources
    ? uniqueSources.map(s => ({ file: s.file, page: s.page, hasLink: true, text: (s as any).text || "" }))
    : citations.map(c => {
        const { file, page } = parseCitation(c);
        // Try to find a matching source from the search results to get the path
        const match = uniqueSources.find(s => s.file.toLowerCase().includes(file.toLowerCase()));
        return { 
          file: match ? match.file : file, 
          page: match ? match.page : (page ? parseInt(page) - 1 : 0), 
          hasLink: !!match || file.toLowerCase().endsWith(".pdf"),
          text: match ? (match as any).text : ""
        };
      });

  if (!displayItems.length) return null;

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
        {displayItems.map((item, i) => (
          <button
            key={i}
            onClick={() => {
              if (item.hasLink) {
                if (onCitationClick) {
                  onCitationClick(item.file, item.page, item.text);
                } else {
                  openPdf(item.file, item.page);
                }
              } else {
                navigator.clipboard.writeText(`${item.file}, Page: ${item.page + 1}`);
              }
            }}
            className="group flex items-center gap-1 px-2 py-0.5 rounded font-mono text-[0.72rem] transition-all duration-300 cursor-pointer hover:scale-105 max-w-full min-w-0"
            style={{
              background: "var(--surface-hover)",
              border: "1px solid var(--border)",
              color: "var(--text-secondary)",
            }}
            title={item.hasLink ? `Open: ${item.file} (Page ${item.page + 1})` : `Click to copy: ${item.file}`}
          >
            <span className="material-icons-round text-[11px] opacity-40 group-hover:opacity-100 group-hover:text-[var(--accent)] transition-all shrink-0">
              {item.hasLink ? "open_in_new" : "description"}
            </span>
            <span className="truncate max-w-[280px]">{item.file.replace(/_/g, " ").replace(/\.pdf$/i, "").slice(0, 40)}{item.file.length > 43 ? "..." : ""}</span>
            <span className="text-[0.6rem] px-1 py-0 rounded bg-[var(--accent-bg)] text-[var(--accent)] border border-[var(--accent-glow)] ml-0.5 shrink-0">
              p.{item.page + 1}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
