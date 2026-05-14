"use client";

import { motion } from "framer-motion";
import { useEffect, useRef } from "react";
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
}

export default function ChatHistory({ messages }: ChatHistoryProps) {
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
                <div className="flex justify-end">
                  <div
                    className="max-w-[75%] px-5 py-3.5 rounded-2xl rounded-br-md text-[0.9rem] leading-[1.6] transition-all duration-300"
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
                <div className="flex gap-4 p-6 border-b border-[var(--border)] group hover:bg-[var(--surface-active)] transition-colors duration-300">
                  <div className="w-10 h-10 rounded-xl bg-[var(--accent-bg)] flex items-center justify-center text-[var(--accent)] shrink-0 shadow-sm">
                    <span className="material-icons-round text-[18px]">psychology</span>
                  </div>
                  <div className="flex-1 min-w-0">
                    {msg.agentThoughts && msg.agentThoughts.length > 0 && (
                      <div className="mb-4">
                        <div className="flex items-start gap-2 text-[0.8rem] text-[var(--text-muted)] p-4 bg-[var(--accent-bg)] border border-[var(--accent-dim)] rounded-xl font-mono leading-relaxed whitespace-pre-wrap max-h-[350px] overflow-y-auto scrollbar-thin scrollbar-thumb-[var(--accent-dim)]">
                          <span className="material-icons-round text-[16px] text-[var(--accent)] mt-0.5 shrink-0">psychology</span>
                          <div className="flex-1">
                            <span className="block text-[0.6rem] font-bold uppercase tracking-widest opacity-40 mb-2 border-b border-[var(--accent-dim)] pb-1">Agentic Reasoning Chain</span>
                            {msg.agentThoughts}
                          </div>
                        </div>
                      </div>
                    )}

                    <p
                      className="text-[0.9rem] leading-[1.8] whitespace-pre-wrap transition-colors duration-300"
                      style={{ color: "var(--text)" }}
                    >
                      {msg.text}
                    </p>
                    
                    {/* Metadata Footer */}
                    {msg.metadata && (
                      <div className="mt-4 flex items-center gap-4 opacity-40 hover:opacity-100 transition-opacity duration-300">
                        <div className="flex items-center gap-1.5">
                          <div className="w-8 h-8 rounded-lg bg-[var(--accent-bg)] flex items-center justify-center text-[var(--accent)] shrink-0">
                            <span className="material-icons-round text-[14px]">psychology</span>
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

                    {(citations.length > 0 || (msg.sources && msg.sources.length > 0)) && 
                      <CitationCard citations={citations} sources={msg.sources} />
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
