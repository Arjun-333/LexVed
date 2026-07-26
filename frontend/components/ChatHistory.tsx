"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import CitationCard from "./CitationCard";

export interface Message {
  id: string;
  text: string;
  isUser: boolean;
  metadata?: {
    retrieval_time?: number;
    generation_time?: number;
    category?: string;
    subcategory?: string;
  };
  sources?: { file: string; page: number; path: string }[];
  agentThoughts?: string;
}

function extractCitations(text: string): string[] {
  const regex = /\[Source:\s*([^\]]+)\]/g;
  const matches: string[] = [];
  let m;
  while ((m = regex.exec(text)) !== null) {
    matches.push(m[1].trim());
  }
  return [...new Set(matches)];
}

interface ChatHistoryProps {
  messages: Message[];
  onCitationClick?: (file: string, page: number, text: string) => void;
}

/** Copy button with ✓ confirmation flash */
function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // fallback for older browsers
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <button
      onClick={handleCopy}
      title={copied ? "Copied!" : "Copy answer"}
      className="flex items-center gap-1 px-2 py-1 rounded-lg text-[0.65rem] font-semibold uppercase tracking-wider transition-all duration-200 cursor-pointer"
      style={{
        background: copied ? "rgba(212,175,55,0.15)" : "rgba(255,255,255,0.05)",
        border: `1px solid ${copied ? "rgba(212,175,55,0.5)" : "rgba(255,255,255,0.08)"}`,
        color: copied ? "#D4AF37" : "#666",
      }}
    >
      <span className="material-icons-round text-[13px]">
        {copied ? "check" : "content_copy"}
      </span>
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

export default function ChatHistory({ messages, onCitationClick }: ChatHistoryProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: "smooth"
      });
    }
  }, [messages]);

  return (
    <div
      ref={scrollRef}
      className="flex-1 overflow-y-auto"
      style={{ maskImage: "linear-gradient(to bottom, transparent, black 1.5%, black 97%, transparent)" }}
    >
      <div className="max-w-[700px] mx-auto py-10 space-y-8">
        {messages.map((msg) => {
          const citations = !msg.isUser ? extractCitations(msg.text) : [];

          return (
            <motion.div
              key={msg.id}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
            >
              {msg.isUser ? (
                /* ── User bubble ──────────────────────────────────── */
                <div className="flex justify-end">
                  <div
                    className="max-w-[75%] px-5 py-3.5 rounded-2xl rounded-br-md text-[0.9rem] leading-[1.6] transition-all duration-300 break-words [overflow-wrap:anywhere]"
                    style={{
                      background: "var(--user-bubble)",
                      border: "1px solid var(--user-bubble-border)",
                      color: "var(--text)",
                      boxShadow: "var(--shadow-prestige)",
                    }}
                  >
                    {msg.text}
                  </div>
                </div>
              ) : (
                /* ── Assistant answer ────────────────────────────── */
                <div className="flex gap-4 group">
                  {/* Avatar */}
                  <div className="w-10 h-10 rounded-xl bg-[var(--accent-bg)] flex items-center justify-center text-[var(--accent)] shrink-0 shadow-sm mt-1">
                    <span className="material-icons-round text-[18px]">psychology</span>
                  </div>

                  <div className="flex-1 min-w-0 space-y-3">
                    {/* Agent thoughts */}
                    {msg.agentThoughts && msg.agentThoughts.length > 0 && (
                      <div className="flex items-start gap-2 text-[0.8rem] text-[var(--text-muted)] p-4 bg-[var(--accent-bg)] border border-[var(--accent-dim)] rounded-xl font-mono leading-relaxed whitespace-pre-wrap max-h-[350px] overflow-y-auto scrollbar-thin scrollbar-thumb-[var(--accent-dim)]">
                        <span className="material-icons-round text-[16px] text-[var(--accent)] mt-0.5 shrink-0">psychology</span>
                        <div className="flex-1">
                          <span className="block text-[0.6rem] font-bold uppercase tracking-widest opacity-40 mb-2 border-b border-[var(--accent-dim)] pb-1">Agentic Reasoning Chain</span>
                          {msg.agentThoughts}
                        </div>
                      </div>
                    )}

                    {/* ── Dark charcoal answer container ────────────────────── */}
                    <div
                      className="relative rounded-2xl p-5 overflow-hidden break-words [overflow-wrap:anywhere]"
                      style={{
                        background: "#161618",
                        border: "1px solid rgba(255, 255, 255, 0.08)",
                        backdropFilter: "blur(12px)",
                        boxShadow: "0 4px 24px rgba(0, 0, 0, 0.45)",
                      }}
                    >
                      {/* Copy button — top-right corner */}
                      <div className="absolute top-3 right-3 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                        <CopyButton text={msg.text} />
                      </div>

                      <p
                        className="text-[0.9rem] leading-[1.8] whitespace-pre-wrap break-words [overflow-wrap:anywhere] pr-16 transition-colors duration-300"
                        style={{ color: "var(--text)" }}
                      >
                        {msg.text}
                      </p>
                    </div>

                    {/* Metadata footer */}
                    {msg.metadata && (
                      <div className="flex items-center gap-4 opacity-40 hover:opacity-100 transition-opacity duration-300 px-1">
                        <div className="flex items-center gap-1.5">
                          <div className="w-7 h-7 rounded-lg bg-[var(--accent-bg)] flex items-center justify-center text-[var(--accent)] shrink-0">
                            <span className="material-icons-round text-[13px]">psychology</span>
                          </div>
                          <span className="text-[0.65rem] font-bold uppercase tracking-widest" style={{ color: "var(--accent)" }}>
                            LexVed Intelligence Engine
                          </span>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <span className="material-icons-round text-[12px]">bolt</span>
                          <span className="text-[0.65rem] font-bold uppercase tracking-wider">
                            Generation: {(msg.metadata.generation_time || 0).toFixed(2)}s
                          </span>
                        </div>
                        {msg.metadata.category && (
                          <div className="flex items-center gap-1.5 ml-auto">
                            <span className="text-[0.65rem] px-2 py-0.5 rounded-full border border-[var(--border)] uppercase tracking-tighter">
                              {msg.metadata.category} • {msg.metadata.subcategory}
                            </span>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Citations */}
                    {(citations.length > 0 || (msg.sources && msg.sources.length > 0)) &&
                      <CitationCard citations={citations} sources={msg.sources} onCitationClick={onCitationClick} />
                    }
                  </div>
                </div>
              )}
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
