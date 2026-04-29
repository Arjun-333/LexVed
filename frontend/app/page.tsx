"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ThemeProvider } from "../components/ThemeProvider";
import Sidebar from "../components/Sidebar";
import WelcomeHero from "../components/WelcomeHero";
import ChatHistory, { Message } from "../components/ChatHistory";
import InputBar from "../components/InputBar";
import ModelWheel from "../components/ModelWheel";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:5000";

interface HealthData {
  active_generation_model?: string;
  ollama?: string;
  active_vector_db?: string;
  uptime_seconds?: number;
}

function Dashboard() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [health, setHealth] = useState<HealthData | null>(null);
  
  // Conversation history for multi-turn chat
  const conversationHistory = useRef<{question: string; answer: string}[]>([]);

  // Fetch health status
  useEffect(() => {
    const fetchHealth = () => {
      fetch(`${API_URL}/api/health`)
        .then(res => res.json())
        .then(data => setHealth(data))
        .catch(() => setHealth({ ollama: "offline" }));
    };
    fetchHealth();
    const interval = setInterval(fetchHealth, 15000); // Poll every 15s
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    let timer: any;
    if (isTyping) {
      setElapsed(0);
      timer = setInterval(() => {
        setElapsed((prev) => prev + 0.1);
      }, 100);
    } else {
      clearInterval(timer);
    }
    return () => clearInterval(timer);
  }, [isTyping]);

  const [hasStarted, setHasStarted] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<{ focus: () => void }>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl+K — Focus search input
      if ((e.ctrlKey || e.metaKey) && e.key === "k") {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  async function handleSend(text: string) {
    if (!hasStarted) setHasStarted(true);

    const userMsg: Message = { id: `u-${Date.now()}`, text, isUser: true };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);
    setIsTyping(true);

    let botMsgId = `b-${Date.now()}`;
    
    try {
      const res = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          message: text,
          history: conversationHistory.current.slice(-3)  // Send last 3 turns
        }),
      });

      if (!res.body) throw new Error("No response body");
      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      // Placeholder for bot message
      setMessages((prev) => [
        ...prev,
        { id: botMsgId, text: "", isUser: false },
      ]);

      let fullText = "";
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n").filter(l => l.trim());

        for (const line of lines) {
          try {
            const data = JSON.parse(line);
            
            if (data.type === "metadata") {
              setMessages((prev) => prev.map(m => m.id === botMsgId ? {
                ...m,
                metadata: {
                   retrieval_time: data.retrieval_time,
                   category: data.category,
                   subcategory: data.subcategory
                }
              } : m));
            } else if (data.type === "content") {
              fullText += data.text;
              setMessages((prev) => prev.map(m => m.id === botMsgId ? { ...m, text: fullText } : m));
              // Once we receive the first bit of content, stop the "Searching" indicator
              setIsTyping(false);
            }
          } catch (e) {
            console.error("Error parsing stream chunk", e);
          }
        }
      }

      // Store in conversation history for multi-turn
      conversationHistory.current.push({ question: text, answer: fullText });
      
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Unknown error";
      setMessages((prev) => [
        ...prev,
        { id: `e-${Date.now()}`, text: `Connection error: ${msg}. Is the backend running?`, isUser: false },
      ]);
    } finally {
      setLoading(false);
      setIsTyping(false);
    }
  }

  // Dynamic status bar values
  const modelName = health?.active_generation_model || "Connecting...";
  const ollamaStatus = health?.ollama === "connected" ? "OPTIMAL" : health?.ollama === "offline" ? "OFFLINE" : "CONNECTING";
  const statusColor = health?.ollama === "connected" ? "bg-[#D4AF37]" : health?.ollama === "offline" ? "bg-[#AA8C2C]/50" : "bg-[#D4AF37]/50";

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: "var(--bg)" }}>
      <Sidebar />
      <main className="flex-1 flex flex-col min-w-0 relative">
        <div className="absolute top-0 left-0 right-0 h-24 pointer-events-none z-10"
             style={{ background: "linear-gradient(to bottom, var(--bg) 0%, transparent 100%)" }} />

        <div className="h-[64px] flex items-center justify-between px-8 shrink-0 z-20">
          <div className="flex flex-col">
             <h1 className="font-bold text-xl tracking-tight transition-colors duration-300"
                 style={{ fontFamily: "var(--font-display)", color: "var(--text)" }}>
               Lex<span style={{ color: "var(--accent)" }}>Ved</span>
             </h1>
          </div>
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex items-center gap-3">
             <div className="flex flex-col items-end">
               <span className="text-[0.6rem] font-bold uppercase tracking-[0.1em] opacity-40">System Status</span>
               <div className="flex items-center gap-2">
                 <span className={`w-1.5 h-1.5 rounded-full ${statusColor} ${health?.ollama === "connected" ? "animate-pulse" : ""}`} />
                 <span className="text-[0.65rem] font-bold uppercase tracking-wider" style={{ color: "var(--text-secondary)" }}>
                   {modelName} · {ollamaStatus}
                 </span>
               </div>
             </div>
          </motion.div>
        </div>

        <div className="flex-1 flex flex-col overflow-hidden relative z-0">
          <AnimatePresence mode="wait">
            {!hasStarted ? (
              <motion.div key="hero" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                exit={{ opacity: 0, y: -20, transition: { duration: 0.3 } }} className="flex-1 flex flex-col">
                <WelcomeHero onSuggestionClick={handleSend} />
              </motion.div>
            ) : (
              <motion.div key="chat" initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                className="flex-1 flex flex-col overflow-hidden">
                <ChatHistory messages={messages} />

                {isTyping && (
                  <div className="max-w-[720px] mx-auto w-full px-6 py-4">
                    <div className="flex gap-4 p-6 rounded-2xl border border-[var(--border)] bg-[var(--surface)] animate-pulse shadow-sm">
                      <div className="w-10 h-10 rounded-xl bg-[var(--accent-bg)] flex items-center justify-center text-[var(--accent)]">
                        <span className="material-icons-round text-[20px] animate-spin">verified_user</span>
                      </div>
                      <div className="flex flex-col gap-2">
                        <div className="text-[var(--text-muted)] text-sm flex items-center gap-3">
                          <span className="font-bold uppercase tracking-widest text-[10px]">Querying Legal Repositories</span>
                          <span className="text-[var(--accent)] font-mono text-[11px] bg-[var(--accent-bg)] px-2 py-0.5 rounded border border-[var(--accent-glow)]">
                            {elapsed < 25 ? (
                              <>ANALYZING CONTEXT | ELAPSED: {elapsed.toFixed(1)}S</>
                            ) : (
                              <>GROUNDING FACTUAL ANALYSIS | {elapsed.toFixed(1)}S</>
                            )}
                          </span>
                        </div>
                        <div className="h-4 w-48 bg-[var(--text-muted)] rounded-full opacity-10" />
                      </div>
                    </div>
                  </div>
                )}
                <div ref={bottomRef} className="h-10 shrink-0" />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <div className="shrink-0 z-20">
           <InputBar onSend={handleSend} disabled={loading} ref={inputRef} />
        </div>
      </main>
      <ModelWheel />
    </div>
  );
}

export default function Home() {
  return (
    <ThemeProvider>
       <Dashboard />
    </ThemeProvider>
  );
}
