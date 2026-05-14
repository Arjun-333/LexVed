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
  // Open PDF in a new tab via the backend API with page fragment
  const token = typeof window !== "undefined" ? localStorage.getItem("lexved_token") : null;
  // Use a direct window.open with the PDF URL — the browser will handle PDF viewing
  if (token) {
    const finalFilename = filename.toLowerCase().endsWith(".pdf") ? filename : `${filename}.pdf`;
    const finalUrl = `${API_URL}/api/pdf/${encodeURIComponent(finalFilename)}#page=${page + 1}`;

    fetch(finalUrl, { headers: { Authorization: `Bearer ${token}` } })
      .then(res => {
        if (!res.ok) throw new Error("PDF not found");
        return res.blob();
      })
      .then(blob => {
        const blobUrl = URL.createObjectURL(blob);
        // Create a temporary link to trigger download/view in a new tab
        const link = document.createElement("a");
        link.href = `${blobUrl}#page=${page + 1}`;
        link.target = "_blank";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      })
      .catch(() => {
        // Fallback: copy citation
        navigator.clipboard.writeText(`${filename}, Page: ${page + 1}`);
      });
  }
}

export default function CitationCard({ citations, sources }: CitationCardProps) {
  // Merge parsed citations with structured sources
  const uniqueSources = sources ? deduplicateSources(sources) : [];
  
  // If we have structured sources, prefer those
  const hasSources = uniqueSources.length > 0;
  const displayItems = hasSources
    ? uniqueSources.map(s => ({ file: s.file, page: s.page, hasLink: true }))
    : citations.map(c => {
        const { file, page } = parseCitation(c);
        // Try to find a matching source from the search results to get the path
        const match = uniqueSources.find(s => s.file.toLowerCase().includes(file.toLowerCase()));
        return { 
          file: match ? match.file : file, 
          page: match ? match.page : (page ? parseInt(page) : 0), 
          hasLink: !!match || file.toLowerCase().endsWith(".pdf") 
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
                openPdf(item.file, item.page);
              } else {
                navigator.clipboard.writeText(`${item.file}, Page: ${item.page}`);
              }
            }}
            className="group flex items-center gap-1 px-2 py-0.5 rounded font-mono text-[0.72rem] transition-all duration-300 cursor-pointer hover:scale-105"
            style={{
              background: "var(--surface-hover)",
              border: "1px solid var(--border)",
              color: "var(--text-secondary)",
            }}
            title={item.hasLink ? `Open: ${item.file} (Page ${item.page + 1})` : `Click to copy: ${item.file}`}
          >
            <span className="material-icons-round text-[11px] opacity-40 group-hover:opacity-100 group-hover:text-[var(--accent)] transition-all">
              {item.hasLink ? "open_in_new" : "description"}
            </span>
            <span>{item.file.replace(/_/g, " ").replace(/\.pdf$/i, "").slice(0, 40)}{item.file.length > 43 ? "..." : ""}</span>
            <span className="text-[0.6rem] px-1 py-0 rounded bg-[var(--accent-bg)] text-[var(--accent)] border border-[var(--accent-glow)] ml-0.5">
              p.{item.page + 1}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
