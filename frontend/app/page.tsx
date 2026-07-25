"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ThemeProvider } from "../components/ThemeProvider";
import { AuthProvider, useAuth } from "../components/AuthContext";
import LoginPage from "../components/LoginPage";
import LandingPage from "../components/LandingPage";
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
  const { authFetch, user } = useAuth();
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [health, setHealth] = useState<HealthData | null>(null);
  const [activeModel, setActiveModel] = useState("ensemble");
  const [activePdf, setActivePdf] = useState<{ filename: string; page: number; text: string } | null>(null);
  const [activeSection, setActiveSection] = useState("Criminal Cases");
  const [viewMode, setViewMode] = useState<"list" | "grid" | "split">("list");
  const [hasStarted, setHasStarted] = useState(false);

  const conversationHistory = useRef<{ question: string; answer: string }[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<{ focus: () => void }>(null);

  const handleCitationClick = useCallback((file: string, page: number, text: string) => {
    setActivePdf({ filename: file, page, text });
  }, []);

  const handleNewBrief = useCallback(() => {
    setMessages([]);
    setHasStarted(false);
    setActivePdf(null);
    conversationHistory.current = [];
    if (typeof window !== "undefined") {
      localStorage.removeItem("lexved_messages");
    }
  }, []);

  // Fetch health status
  useEffect(() => {
    const fetchHealth = () => {
      authFetch(`${API_URL}/api/health`)
        .then(res => res.json())
        .then(data => {
          setHealth(data);
          if (data.active_generation_model) {
            setActiveModel(data.active_generation_model);
          }
        })
        .catch(() => setHealth({ ollama: "offline" }));
    };
    fetchHealth();
    const interval = setInterval(fetchHealth, 10000);
    return () => clearInterval(interval);
  }, [authFetch]);

  const handleModelChange = useCallback((newModel: string) => {
    setActiveModel(newModel);
  }, []);

  useEffect(() => {
    let timer: ReturnType<typeof setInterval>;
    if (isTyping) {
      setElapsed(0);
      timer = setInterval(() => {
        setElapsed((prev) => prev + 0.1);
      }, 100);
    } else {
      clearInterval(timer!);
    }
    return () => clearInterval(timer!);
  }, [isTyping]);

  // (hasStarted, bottomRef, inputRef declared above)

  useEffect(() => {
    setTimeout(() => {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, 50);
  }, [messages, isTyping]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "k") {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  const abortController = useRef<AbortController | null>(null);

  async function handleStop() {
    if (abortController.current) {
      abortController.current.abort();
      abortController.current = null;
      setLoading(false);
      setIsTyping(false);
    }
  }

  async function handleSend(text: string, agentic: boolean = false) {
    if (!hasStarted) setHasStarted(true);

    if (abortController.current) {
      abortController.current.abort();
    }
    abortController.current = new AbortController();

    const userMsg: Message = { id: `u-${Date.now()}`, text, isUser: true };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);
    setIsTyping(true);

    let botMsgId = `b-${Date.now()}`;

    try {
      const res = await authFetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: abortController.current.signal,
        body: JSON.stringify({
          message: text,
          history: conversationHistory.current.slice(-3),
          agentic
        }),
      });

      if (!res.body) throw new Error("No response body");
      const reader = res.body.getReader();
      const decoder = new TextDecoder();

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
                },
                sources: data.sources || []
              } : m));
            } else if (data.type === "agent_thought") {
              setMessages((prev) => prev.map(m => m.id === botMsgId ? {
                ...m,
                agentThoughts: (m.agentThoughts || "") + data.text
              } : m));
            } else if (data.type === "content") {
              fullText += data.text;
              setMessages((prev) => prev.map(m => m.id === botMsgId ? { ...m, text: fullText } : m));
              setIsTyping(false);
            } else if (data.type === "done") {
              setMessages((prev) => prev.map(m => m.id === botMsgId ? {
                ...m,
                metadata: { ...m.metadata, generation_time: data.generation_time }
              } : m));
            }
          } catch (e) {
            console.error("Error parsing stream chunk", e);
          }
        }
      }

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

  async function handleUpload(file: File) {
    if (!hasStarted) setHasStarted(true);

    const botMsgId = `b-upload-${Date.now()}`;
    setLoading(true);
    setIsTyping(true);

    setMessages((prev) => [
      ...prev,
      {
        id: botMsgId,
        text: `Uploading and preparing "${file.name}" for vector indexing...`,
        isUser: false
      },
    ]);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await authFetch(`${API_URL}/api/ingest`, {
        method: "POST",
        body: formData,
      });

      if (!res.body) throw new Error("No response body");
      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      let lastStep = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n").filter(l => l.trim());

        for (const line of lines) {
          try {
            const data = JSON.parse(line);
            if (data.step) {
              lastStep = data.step;
              setMessages((prev) => prev.map(m => m.id === botMsgId ? {
                ...m,
                text: lastStep,
                agentThoughts: (m.agentThoughts || "") + `[Ingestion Pipeline] ${lastStep}\n`
              } : m));
            }
          } catch (e) {}
        }
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Unknown error";
      setMessages((prev) => prev.map(m => m.id === botMsgId ? {
        ...m,
        text: `Upload failed: ${msg}. Check backend server logs.`
      } : m));
    } finally {
      setLoading(false);
      setIsTyping(false);
    }
  }

  // Dynamic model display names from backend config
  const [modelDisplayNames, setModelDisplayNames] = useState<Record<string, string>>({});

  useEffect(() => {
    authFetch(`${API_URL}/api/settings/config`)
      .then(res => res.json())
      .then(data => {
        if (data.generation_model_metadata) {
          const names: Record<string, string> = {};
          data.generation_model_metadata.forEach((m: { id: string; name: string }) => {
            names[m.id] = m.name;
          });
          setModelDisplayNames(names);
        }
      })
      .catch(() => {});
  }, [authFetch]);

  const modelName = modelDisplayNames[activeModel] || activeModel || "Connecting...";
  const ollamaStatus = health?.ollama === "connected" ? "CONNECTING" : health?.ollama === "offline" ? "OFFLINE" : "CONNECTING";
  const isOllamaConnected = health?.ollama === "connected";

  const displayName = user?.displayName || user?.username || "Counsel";
  const userInitial = displayName.charAt(0).toUpperCase();

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: "#0c0c0c" }}>
      {/* Sidebar */}
      <Sidebar
        onNewBrief={handleNewBrief}
        activeSection={activeSection}
        onSectionChange={setActiveSection}
      />

      {/* Main area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Canvas + Chat Panel */}
        <main
          className={`flex flex-col min-w-0 relative h-full transition-all duration-500 ease-[cubic-bezier(0.16,1,0.3,1)] ${activePdf ? "w-3/5" : "w-full"}`}
        >
          {/* Top gradient fade */}
          <div
            className="absolute top-0 left-0 right-0 h-20 pointer-events-none z-10"
            style={{ background: "linear-gradient(to bottom, #0c0c0c 0%, transparent 100%)" }}
          />

          {/* ===== HEADER BAR ===== */}
          <div
            className="h-[54px] flex items-center justify-between px-6 shrink-0 z-20 relative"
            style={{ borderBottom: "1px solid #181818" }}
          >
            {/* Left: Section title */}
            <div className="flex items-center gap-2">
              <h1
                className="font-bold text-[1rem]"
                style={{
                  color: "#ffffff",
                  fontFamily: "var(--font-body)",
                  letterSpacing: "-0.025em",
                  fontWeight: 700,
                }}
              >
                {hasStarted
                  ? (messages.find(m => m.isUser)?.text?.slice(0, 40) || activeSection) +
                    (messages.find(m => m.isUser)?.text?.length! > 40 ? "…" : "")
                  : activeSection}
              </h1>
            </div>

            {/* Center: Status Badges */}
            <div className="flex items-center gap-4 absolute left-1/2 -translate-x-1/2">
              {/* System Status */}
              <div className="flex flex-col items-center gap-0.5">
                <span
                  className="text-[0.5rem] font-bold uppercase tracking-[0.12em]"
                  style={{ color: "#444" }}
                >
                  System Status
                </span>
                <div className="flex items-center gap-1.5">
                  <span
                    className="w-1.5 h-1.5 rounded-full"
                    style={{
                      background: isOllamaConnected ? "#D4AF37" : "#555",
                      boxShadow: isOllamaConnected ? "0 0 6px #D4AF37" : "none",
                      animation: isOllamaConnected ? "pulse-dot 2s infinite" : "none",
                    }}
                  />
                  <span
                    className="text-[0.58rem] font-bold uppercase tracking-[0.08em]"
                    style={{ color: "#666" }}
                  >
                    ENSEMBLE · {ollamaStatus}
                  </span>
                </div>
              </div>

              {/* Separator */}
              <div className="w-px h-6" style={{ background: "#222" }} />

              {/* Intelligence Engine */}
              <div className="flex flex-col items-center gap-0.5">
                <span
                  className="text-[0.5rem] font-bold uppercase tracking-[0.12em]"
                  style={{ color: "#444" }}
                >
                  Intelligence Engine
                </span>
                <span
                  className="text-[0.58rem] font-bold uppercase tracking-[0.08em]"
                  style={{ color: "#666" }}
                >
                  LOCAL CLUSTER STATUS: {isOllamaConnected ? "OPTIMAL" : "OFFLINE"}
                </span>
              </div>
            </div>

            {/* Right: User pill + View toggles + More */}
            <div className="flex items-center gap-2">
              {/* User pill */}
              <div
                className="flex items-center gap-2 px-3 py-1.5 rounded-full"
                style={{
                  background: "rgba(255,255,255,0.05)",
                  border: "1px solid rgba(255,255,255,0.08)",
                }}
              >
                <div
                  className="w-5 h-5 rounded-full flex items-center justify-center text-[9px] font-bold shrink-0"
                  style={{
                    background: "linear-gradient(135deg, #2a2a2a, #1a1800)",
                    color: "#D4AF37",
                    border: "1px solid rgba(212,175,55,0.2)",
                  }}
                >
                  {userInitial}
                </div>
                <span
                  className="text-[0.72rem] font-semibold"
                  style={{ color: "#cccccc", letterSpacing: "-0.01em" }}
                >
                  {displayName}
                </span>
              </div>

              {/* View toggle group */}
              <div
                className="flex items-center rounded-lg p-0.5 gap-0.5"
                style={{ background: "rgba(255,255,255,0.04)", border: "1px solid #1e1e1e" }}
              >
                {(["list", "grid", "split"] as const).map((mode, idx) => {
                  const icons = ["format_list_bulleted", "grid_view", "view_sidebar"];
                  return (
                    <button
                      key={mode}
                      onClick={() => setViewMode(mode)}
                      className="view-toggle-btn"
                      style={{
                        background: viewMode === mode ? "rgba(212,175,55,0.12)" : "transparent",
                        color: viewMode === mode ? "#D4AF37" : "#444",
                      }}
                      title={mode}
                    >
                      <span className="material-icons-round text-[14px]">{icons[idx]}</span>
                    </button>
                  );
                })}
              </div>

              {/* More menu */}
              <button
                className="view-toggle-btn"
                title="More options"
                style={{ color: "#444" }}
              >
                <span className="material-icons-round text-[16px]">more_horiz</span>
              </button>
            </div>
          </div>

          {/* ===== MAIN CONTENT AREA with DOT GRID ===== */}
          <div className="flex-1 flex flex-col overflow-hidden relative dot-grid z-0">
            {/* Radial vignette overlay on the grid */}
            <div
              className="absolute inset-0 pointer-events-none z-0"
              style={{
                background: "radial-gradient(ellipse at center, transparent 30%, #0c0c0c 100%)",
              }}
            />

            <AnimatePresence mode="wait">
              {!hasStarted ? (
                <motion.div
                  key="hero"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0, y: -20, transition: { duration: 0.3 } }}
                  className="flex-1 flex flex-col relative z-10"
                >
                  <WelcomeHero onSuggestionClick={handleSend} />
                </motion.div>
              ) : (
                <motion.div
                  key="chat"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex-1 flex flex-col overflow-hidden relative z-10"
                >
                  <ChatHistory messages={messages} onCitationClick={handleCitationClick} />

                  {isTyping && (
                    <div className="max-w-[720px] mx-auto w-full px-6 py-3">
                      <div
                        className="flex gap-3 p-4 rounded-xl border animate-pulse"
                        style={{
                          background: "#111111",
                          border: "1px solid #1e1e1e",
                        }}
                      >
                        <div
                          className="w-9 h-9 rounded-lg flex items-center justify-center shrink-0"
                          style={{ background: "rgba(212,175,55,0.08)", color: "#D4AF37" }}
                        >
                          <span className="material-icons-round text-[18px] animate-spin">verified_user</span>
                        </div>
                        <div className="flex flex-col gap-2 justify-center">
                          <div className="flex items-center gap-3 text-sm">
                            <span
                              className="font-bold uppercase tracking-widest text-[10px]"
                              style={{ color: "#555" }}
                            >
                              Querying Legal Repositories
                            </span>
                            <span
                              className="font-mono text-[10px] px-2 py-0.5 rounded border"
                              style={{
                                color: "#D4AF37",
                                background: "rgba(212,175,55,0.06)",
                                borderColor: "rgba(212,175,55,0.2)",
                              }}
                            >
                              {elapsed < 25 ? (
                                <>ANALYZING CONTEXT | {elapsed.toFixed(1)}S</>
                              ) : (
                                <>GROUNDING FACTUAL ANALYSIS | {elapsed.toFixed(1)}S</>
                              )}
                            </span>
                          </div>
                          <div className="h-3 w-40 bg-[#222] rounded-full opacity-60" />
                        </div>
                      </div>
                    </div>
                  )}
                  <div ref={bottomRef} className="h-8 shrink-0" />
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* ===== INPUT BAR ===== */}
          <div className="shrink-0 z-20 pb-2" style={{ background: "#0c0c0c" }}>
            <InputBar
              onSend={handleSend}
              onStop={handleStop}
              onUpload={handleUpload}
              disabled={loading}
              userInitial={userInitial}
              ref={inputRef}
            />
          </div>
        </main>

        {/* PDF Viewer Panel */}
        {activePdf && (
          <aside className="w-2/5 flex flex-col h-full relative overflow-hidden transition-all duration-500 ease-[cubic-bezier(0.16,1,0.3,1)]"
            style={{ background: "#0f0f0f", borderLeft: "1px solid #1e1e1e" }}
          >
            <div
              className="h-[54px] flex items-center justify-between px-5 shrink-0"
              style={{ borderBottom: "1px solid #1e1e1e" }}
            >
              <span
                className="font-bold text-xs uppercase tracking-widest truncate max-w-[80%] flex items-center gap-2"
                style={{ color: "#aaaaaa" }}
              >
                <span className="material-icons-round text-[15px]" style={{ color: "#D4AF37" }}>menu_book</span>
                {activePdf.filename.replace(/\.pdf$/i, "").slice(0, 30)} (Page {activePdf.page + 1})
              </span>
              <button
                onClick={() => setActivePdf(null)}
                className="w-7 h-7 rounded-full flex items-center justify-center hover:bg-[#1e1e1e] transition-colors"
                style={{ color: "#555" }}
              >
                <span className="material-icons-round text-[16px]">close</span>
              </button>
            </div>

            <div className="flex-1 bg-[#000] relative overflow-hidden">
              <iframe
                src={`${API_URL}/api/pdf/${encodeURIComponent(activePdf.filename)}?token=${localStorage.getItem("lexved_token")}#page=${activePdf.page + 1}`}
                className="w-full h-full border-none"
              />
            </div>

            {activePdf.text && (
              <div
                className="h-[160px] shrink-0 p-4 overflow-y-auto"
                style={{
                  borderTop: "1px solid #1e1e1e",
                  background: "#0f0f0f",
                  borderLeft: "3px solid #D4AF37",
                }}
              >
                <div className="flex items-center gap-1.5 mb-2 text-[0.6rem] font-bold uppercase tracking-widest" style={{ color: "#D4AF37" }}>
                  <span className="material-icons-round text-[13px]">highlight</span>
                  Highlighted Match Context
                </div>
                <p className="text-[0.75rem] leading-[1.6] italic font-mono whitespace-pre-wrap" style={{ color: "#888" }}>
                  &quot;{activePdf.text}&quot;
                </p>
              </div>
            )}
          </aside>
        )}

        {/* Right Panel: Model Cards */}
        <ModelWheel activeModelId={activeModel} onModelChange={handleModelChange} />
      </div>
    </div>
  );
}

function AuthGate() {
  const { isAuthenticated, isLoading } = useAuth();
  const [showLanding, setShowLanding] = useState(true);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ background: "#000" }}>
        <div className="flex flex-col items-center gap-4">
          <div className="w-9 h-9 border-2 border-[#D4AF37]/20 border-t-[#D4AF37] rounded-full animate-spin" />
          <p className="text-[#D4AF37]/40 text-[0.6rem] uppercase tracking-widest font-bold">
            Restoring Session...
          </p>
        </div>
      </div>
    );
  }

  if (isAuthenticated) return <Dashboard />;

  if (showLanding) {
    return <LandingPage onEnter={() => setShowLanding(false)} />;
  }

  return <LoginPage />;
}

export default function Home() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <AuthGate />
      </AuthProvider>
    </ThemeProvider>
  );
}
