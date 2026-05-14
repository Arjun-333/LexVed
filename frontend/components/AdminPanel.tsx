"use client";
import React, { useState, useEffect } from "react";
import { X, Users, Activity, HardDrive, Trash2, FileText, Shield, RefreshCw, Server, Loader2 } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useAuth } from "./AuthContext";

interface SystemInfo { uptime_seconds?: number; ollama?: string; vector_db?: string; active_embedding_model?: string; active_generation_model?: string; collections?: string[]; vector_count?: number; ollama_models?: string[]; }
interface UserInfo { username: string; role: string; display_name: string; }

export default function AdminPanel({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const { authFetch } = useAuth();
  const [health, setHealth] = useState<SystemInfo | null>(null);
  const [users, setUsers] = useState<UserInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeSection, setActiveSection] = useState<"system" | "users" | "data">("system");
  const [actionStatus, setActionStatus] = useState<string | null>(null);
  const API = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:5000";

  useEffect(() => {
    if (!isOpen) return;
    setLoading(true);
    Promise.all([
      authFetch(`${API}/api/health`).then(r => r.json()).catch(() => null),
      authFetch(`${API}/api/admin/users`).then(r => r.json()).catch(() => []),
    ]).then(([h, u]) => { setHealth(h); setUsers(Array.isArray(u) ? u : u?.users || []); setLoading(false); });
    const poll = setInterval(() => {
      authFetch(`${API}/api/health`).then(r => r.json()).then(setHealth).catch(() => {});
    }, 10000);
    return () => clearInterval(poll);
  }, [isOpen, authFetch, API]);

  const showAction = (msg: string) => { setActionStatus(msg); setTimeout(() => setActionStatus(null), 3000); };

  const clearHistory = async () => {
    await authFetch(`${API}/api/history`, { method: "DELETE" });
    showAction("History cleared successfully");
  };

  const clearEvalResults = async () => {
    await authFetch(`${API}/api/admin/clear_eval`, { method: "POST" });
    showAction("Evaluation results cleared");
  };

  const formatUptime = (s: number) => {
    const h = Math.floor(s / 3600); const m = Math.floor((s % 3600) / 60);
    return h > 0 ? `${h}h ${m}m` : `${m}m ${Math.floor(s % 60)}s`;
  };

  if (!isOpen) return null;

  const sections = [
    { id: "system" as const, icon: Server, label: "System" },
    { id: "users" as const, icon: Users, label: "Users" },
    { id: "data" as const, icon: HardDrive, label: "Data" },
  ];

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/80 backdrop-blur-md p-4">
        <motion.div initial={{ opacity: 0, scale: 0.9, y: 30 }} animate={{ opacity: 1, scale: 1, y: 0 }} exit={{ opacity: 0, scale: 0.9, y: 30 }}
          className="w-full max-w-3xl bg-[#0a0a0a] border border-[#D4AF37]/20 rounded-2xl shadow-[0_0_50px_rgba(212,175,55,0.1)] overflow-hidden flex flex-col max-h-[85vh]">

          {/* Header */}
          <div className="p-6 border-b border-[#D4AF37]/20 flex items-center justify-between bg-gradient-to-r from-[#D4AF37]/10 to-transparent">
            <div>
              <h2 className="font-display text-xl text-[#D4AF37] tracking-[0.15em] flex items-center gap-3">
                <Shield className="w-5 h-5" /> Administration Console
              </h2>
              <p className="text-white/40 text-[10px] uppercase font-mono tracking-widest mt-2">System Management · Admin Access Only</p>
            </div>
            <button onClick={onClose} className="p-2 hover:bg-white/5 rounded-full transition-colors"><X className="text-white/40 hover:text-white" /></button>
          </div>

          {/* Action Status */}
          <AnimatePresence>
            {actionStatus && (
              <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }}
                className="px-6 py-2 bg-[#D4AF37]/10 border-b border-[#D4AF37]/20 text-[#D4AF37] text-xs font-bold uppercase tracking-wider text-center">{actionStatus}</motion.div>
            )}
          </AnimatePresence>

          {/* Section Tabs */}
          <div className="flex border-b border-white/5 px-6">
            {sections.map(s => (
              <button key={s.id} onClick={() => setActiveSection(s.id)}
                className={`flex items-center gap-2 px-4 py-3 text-[10px] font-bold uppercase tracking-[0.15em] border-b-2 transition-all ${activeSection === s.id ? "border-[#D4AF37] text-[#D4AF37]" : "border-transparent text-white/30 hover:text-white/60"}`}>
                <s.icon className="w-3.5 h-3.5" /> {s.label}
              </button>
            ))}
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto p-6">
            {loading ? (
              <div className="flex flex-col items-center justify-center py-20 gap-4">
                <Loader2 className="w-10 h-10 text-[#D4AF37] animate-spin" /><p className="text-white/30 text-xs uppercase tracking-widest">Loading System Data...</p>
              </div>
            ) : activeSection === "system" ? (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  {[
                    { label: "Uptime", val: health?.uptime_seconds ? formatUptime(health.uptime_seconds) : "—", icon: Activity },
                    { label: "Ollama", val: health?.ollama === "connected" ? "CONNECTED" : "OFFLINE", icon: Server },
                    { label: "Vector DB", val: health?.vector_db === "connected" ? "CONNECTED" : "OFFLINE", icon: HardDrive },
                    { label: "Gen Model", val: health?.active_generation_model || "—", icon: Activity },
                  ].map((card, i) => (
                    <div key={i} className="p-4 bg-white/[0.03] border border-white/5 rounded-xl group hover:border-[#D4AF37]/20 transition-all">
                      <div className="flex items-center gap-2 mb-2"><card.icon className="w-3.5 h-3.5 text-[#D4AF37]/50" /><span className="text-[9px] uppercase tracking-[0.2em] text-white/30 font-bold">{card.label}</span></div>
                      <div className="text-white text-sm font-bold">{card.val}</div>
                    </div>
                  ))}
                </div>
                <div className="p-4 bg-white/[0.02] border border-white/5 rounded-xl">
                  <div className="text-[9px] uppercase tracking-[0.2em] text-white/30 font-bold mb-3">Embedding Model</div>
                  <div className="text-[#D4AF37] text-sm font-mono">{health?.active_embedding_model || "—"}</div>
                </div>
                {health?.ollama_models && (
                  <div className="p-4 bg-white/[0.02] border border-white/5 rounded-xl">
                    <div className="text-[9px] uppercase tracking-[0.2em] text-white/30 font-bold mb-3">Available Ollama Models</div>
                    <div className="flex flex-wrap gap-2">{health.ollama_models.map(m => (
                      <span key={m} className="text-[10px] font-mono bg-white/5 px-2 py-1 rounded border border-white/10 text-white/60">{m}</span>
                    ))}</div>
                  </div>
                )}
                {health?.collections && (
                  <div className="p-4 bg-white/[0.02] border border-white/5 rounded-xl">
                    <div className="text-[9px] uppercase tracking-[0.2em] text-white/30 font-bold mb-3">Qdrant Collections</div>
                    <div className="flex flex-wrap gap-2">{health.collections.map(c => (
                      <span key={c} className="text-[10px] font-mono bg-[#D4AF37]/5 px-2 py-1 rounded border border-[#D4AF37]/20 text-[#D4AF37]">{c}</span>
                    ))}</div>
                  </div>
                )}
              </div>
            ) : activeSection === "users" ? (
              <div className="space-y-3">
                {users.map((u, i) => (
                  <motion.div key={u.username} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.05 }}
                    className="flex items-center gap-4 p-4 rounded-xl border border-white/5 bg-white/[0.02] hover:border-[#D4AF37]/20 transition-all">
                    <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${u.role === "admin" ? "bg-[#D4AF37]/10 text-[#D4AF37]" : "bg-white/5 text-white/40"}`}>
                      <span className="material-icons-round text-[20px]">{u.role === "admin" ? "admin_panel_settings" : "person"}</span>
                    </div>
                    <div className="flex-1">
                      <div className="text-sm font-medium text-white">{u.display_name}</div>
                      <div className="text-[10px] text-white/30 uppercase tracking-wider font-mono">@{u.username}</div>
                    </div>
                    <span className={`text-[9px] font-bold uppercase tracking-[0.15em] px-3 py-1 rounded-full border ${u.role === "admin" ? "bg-[#D4AF37]/10 text-[#D4AF37] border-[#D4AF37]/30" : "bg-white/5 text-white/40 border-white/10"}`}>{u.role}</span>
                  </motion.div>
                ))}
              </div>
            ) : (
              <div className="space-y-4">
                <div className="text-[9px] uppercase tracking-[0.2em] text-white/30 font-bold mb-2">Data Management</div>
                {[
                  { label: "Clear Query History", desc: "Remove all stored query logs from history.json", icon: Trash2, action: clearHistory, color: "text-orange-400" },
                  { label: "Clear Evaluation Results", desc: "Reset evaluation_results.json and comparative data", icon: RefreshCw, action: clearEvalResults, color: "text-red-400" },
                ].map((item, i) => (
                  <div key={i} className="flex items-center gap-4 p-4 rounded-xl border border-white/5 bg-white/[0.02] hover:border-white/10 transition-all">
                    <div className="w-10 h-10 rounded-xl bg-white/5 flex items-center justify-center"><item.icon className={`w-4 h-4 ${item.color}`} /></div>
                    <div className="flex-1">
                      <div className="text-sm font-medium text-white">{item.label}</div>
                      <div className="text-[10px] text-white/30">{item.desc}</div>
                    </div>
                    <button onClick={item.action} className="text-[9px] font-bold uppercase tracking-[0.15em] px-4 py-2 rounded-lg border border-white/10 text-white/50 hover:bg-red-500/10 hover:border-red-500/30 hover:text-red-400 transition-all">Execute</button>
                  </div>
                ))}

                <div className="mt-6 p-4 bg-white/[0.02] border border-white/5 rounded-xl">
                  <div className="text-[9px] uppercase tracking-[0.2em] text-white/30 font-bold mb-3 flex items-center gap-2"><FileText className="w-3 h-3" /> System Files</div>
                  <div className="grid grid-cols-2 gap-2 text-[10px] font-mono text-white/40">
                    {["system_config.json", "history.json", "evaluation_results.json", "evaluation_data.json", "comparative_results.json", "primitive_evaluation_results.json"].map(f => (
                      <div key={f} className="flex items-center gap-2 py-1"><span className="w-1.5 h-1.5 rounded-full bg-[#D4AF37]/30" />{f}</div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="p-4 bg-black/40 border-t border-white/5 flex items-center justify-between">
            <div className="flex items-center gap-2 text-[10px] text-white/20 uppercase tracking-widest"><Shield className="w-3 h-3 text-[#D4AF37]/50" /> Administrator Session</div>
            <div className="text-[10px] text-white/20 font-mono">{new Date().toLocaleString()}</div>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
