"use client";
import React, { useState, useEffect } from "react";
import { X, Settings, Database, Cpu, Zap } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useAuth } from "./AuthContext";

export default function SettingsModal({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const { authFetch, isAdmin } = useAuth();
  const [embeddingModel, setEmbeddingModel] = useState("");
  const [genModel, setGenModel] = useState("");
  const [vectorDb, setVectorDb] = useState("");
  const [availableEmbeddings, setAvailableEmbeddings] = useState<string[]>([]);
  const [availableGenModels, setAvailableGenModels] = useState<string[]>([]);
  const [availableProviders, setAvailableProviders] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [saveStatus, setSaveStatus] = useState<string | null>(null);
  const API = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:5000";

  useEffect(() => {
    if (!isOpen) return;
    setLoading(true);
    Promise.all([
      authFetch(`${API}/api/settings/embedding_model`).then(r => r.json()).catch(() => ({ model: "" })),
      authFetch(`${API}/api/settings/generation_model`).then(r => r.json()).catch(() => ({ model: "" })),
      authFetch(`${API}/api/settings/vector_db`).then(r => r.json()).catch(() => ({ db: "" })),
      authFetch(`${API}/api/settings/config`).then(r => r.json()).catch(() => ({ embedding_models: [], generation_models: [], providers: [] })),
    ]).then(([emb, gen, db, config]) => {
      setEmbeddingModel(emb.model || "");
      setGenModel(gen.model || "");
      setVectorDb(db.db || "");
      setAvailableEmbeddings(config.embedding_models || []);
      setAvailableGenModels(config.generation_models || []);
      setAvailableProviders(config.providers || []);
      setLoading(false);
    });
  }, [isOpen, authFetch, API]);

  const flash = (msg: string) => { setSaveStatus(msg); setTimeout(() => setSaveStatus(null), 2000); };

  const saveEmbedding = async (model: string) => {
    setEmbeddingModel(model);
    await authFetch(`${API}/api/settings/embedding_model`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model }) });
    flash("Embedding model updated");
  };
  const saveGen = async (model: string) => {
    setGenModel(model);
    await authFetch(`${API}/api/settings/generation_model`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model }) });
    flash("Generation model updated");
  };
  const saveDb = async (db: string) => {
    setVectorDb(db);
    await authFetch(`${API}/api/settings/vector_db`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ db }) });
    flash("Vector DB updated");
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/80 backdrop-blur-md p-4">
        <motion.div initial={{ opacity: 0, scale: 0.9, y: 30 }} animate={{ opacity: 1, scale: 1, y: 0 }} exit={{ opacity: 0, scale: 0.9, y: 30 }}
          className="w-full max-w-xl rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[85vh]"
          style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)' }}>

          <div className="p-6 flex items-center justify-between" style={{ borderBottom: '1px solid var(--border)', background: 'linear-gradient(to right, var(--accent-bg), transparent)' }}>
            <div>
              <h2 className="font-display text-xl tracking-[0.15em] flex items-center gap-3" style={{ color: 'var(--accent)' }}><Settings className="w-5 h-5" /> System Settings</h2>
              <p className="text-[10px] uppercase font-mono tracking-widest mt-2" style={{ color: 'var(--text-muted)' }}>Model & Provider Configuration</p>
            </div>
            <button onClick={onClose} className="p-2 rounded-full transition-colors" style={{ color: 'var(--text-muted)' }}><X /></button>
          </div>

          <AnimatePresence>
            {saveStatus && (
              <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }}
                className="px-6 py-2 bg-[#D4AF37]/10 border-b border-[#D4AF37]/20 text-[#D4AF37] text-xs font-bold uppercase tracking-wider text-center">{saveStatus}</motion.div>
            )}
          </AnimatePresence>

          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            {loading ? (
              <div className="flex flex-col items-center justify-center py-16 gap-4">
                <div className="w-8 h-8 border-2 border-[#D4AF37]/30 border-t-[#D4AF37] rounded-full animate-spin" />
                <p className="text-xs uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>Loading Configuration...</p>
              </div>
            ) : (
              <>
                {/* Generation Model */}
                <div>
                  <div className="flex items-center gap-2 mb-3"><Cpu className="w-3.5 h-3.5" style={{ color: 'var(--accent-dim)' }} /><span className="text-[9px] uppercase tracking-[0.2em] font-bold" style={{ color: 'var(--text-muted)' }}>Generation Model (LLM)</span></div>
                  <div className="grid gap-2">
                    {availableGenModels.map(m => (
                      <button key={m} onClick={() => saveGen(m)}
                        className={`flex items-center gap-3 p-3 rounded-xl text-left transition-all border`}
                        style={{ background: genModel === m ? 'var(--accent-bg)' : 'var(--surface)', borderColor: genModel === m ? 'var(--accent-dim)' : 'var(--border)' }}>
                        <span className="w-2 h-2 rounded-full" style={{ background: genModel === m ? 'var(--accent)' : 'var(--border)' }} />
                        <span className="text-xs font-mono" style={{ color: genModel === m ? 'var(--accent)' : 'var(--text-secondary)', fontWeight: genModel === m ? 700 : 400 }}>{m}</span>
                        {m.includes("instant") && <span className="text-[8px] bg-[#D4AF37]/10 text-[#D4AF37] px-1.5 py-0.5 rounded border border-[#D4AF37]/20 font-bold uppercase">Fast</span>}
                        {m.includes("70b") && <span className="text-[8px] bg-white/5 text-white/40 px-1.5 py-0.5 rounded border border-white/10 font-bold uppercase">Heavy</span>}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Embedding Model */}
                <div>
                  <div className="flex items-center gap-2 mb-3"><Zap className="w-3.5 h-3.5" style={{ color: 'var(--accent-dim)' }} /><span className="text-[9px] uppercase tracking-[0.2em] font-bold" style={{ color: 'var(--text-muted)' }}>Embedding Model</span></div>
                  <div className="grid gap-2">
                    {availableEmbeddings.map(m => (
                      <button key={m} onClick={() => saveEmbedding(m)}
                        className={`flex items-center gap-3 p-3 rounded-xl text-left transition-all border`}
                        style={{ background: embeddingModel === m ? 'var(--accent-bg)' : 'var(--surface)', borderColor: embeddingModel === m ? 'var(--accent-dim)' : 'var(--border)' }}>
                        <span className="w-2 h-2 rounded-full" style={{ background: embeddingModel === m ? 'var(--accent)' : 'var(--border)' }} />
                        <span className="text-xs font-mono" style={{ color: embeddingModel === m ? 'var(--accent)' : 'var(--text-secondary)', fontWeight: embeddingModel === m ? 700 : 400 }}>{m}</span>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Vector Database */}
                <div>
                  <div className="flex items-center gap-2 mb-3"><Database className="w-3.5 h-3.5" style={{ color: 'var(--accent-dim)' }} /><span className="text-[9px] uppercase tracking-[0.2em] font-bold" style={{ color: 'var(--text-muted)' }}>Vector Database Provider</span></div>
                  <div className="grid grid-cols-2 gap-3">
                    {availableProviders.map(p => (
                      <button key={p} onClick={() => saveDb(p)}
                        className={`p-4 rounded-xl text-center transition-all border`}
                        style={{ background: vectorDb === p ? 'var(--accent-bg)' : 'var(--surface)', borderColor: vectorDb === p ? 'var(--accent-dim)' : 'var(--border)' }}>
                        <Database className="w-5 h-5 mx-auto mb-2" style={{ color: vectorDb === p ? 'var(--accent)' : 'var(--text-muted)' }} />
                        <span className="text-xs font-bold uppercase tracking-wider" style={{ color: vectorDb === p ? 'var(--accent)' : 'var(--text-muted)' }}>{p}</span>
                      </button>
                    ))}
                  </div>
                </div>
              </>
            )}
          </div>

          <div className="p-4 flex items-center justify-center" style={{ background: 'var(--surface)', borderTop: '1px solid var(--border)' }}>
            <p className="text-[10px] uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>Changes take effect immediately</p>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
