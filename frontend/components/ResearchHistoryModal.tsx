"use client";

import React, { useState, useEffect } from "react";
import { X, History, Search, Clock, CheckCircle2, AlertCircle } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useAuth } from "./AuthContext";

interface HistoryItem {
  id: number;
  query: string;
  date: string;
  status: string;
  metrics?: {
    retrieval_lat: number;
    e2e_lat: number;
    ans_length: number;
  };
}

export default function ResearchHistoryModal({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const { authFetch } = useAuth();
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:5000";

  useEffect(() => {
    if (isOpen) {
      setLoading(true);
      authFetch(`${API_URL}/api/history`)
        .then((res) => res.json())
        .then((data) => { setHistory(data); setLoading(false); })
        .catch((err) => { console.error(err); setLoading(false); });
    }
  }, [isOpen, authFetch, API_URL]);

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/80 backdrop-blur-md p-4">
        <motion.div 
          initial={{ opacity: 0, scale: 0.9, y: 30 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.9, y: 30 }}
          className="w-full max-w-2xl rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[80vh]"
          style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)' }}
        >
          <div className="p-6 flex items-center justify-between" style={{ borderBottom: '1px solid var(--border)', background: 'linear-gradient(to right, var(--accent-bg), transparent)' }}>
            <div>
              <h2 className="font-display text-xl text-[#D4AF37] tracking-[0.15em] flex items-center gap-3">
                <History className="text-[#D4AF37] w-5 h-5" />
                Research History
              </h2>
              <p className="text-[10px] uppercase font-mono tracking-widest mt-2" style={{ color: 'var(--text-muted)' }}>Audit log of institutional queries</p>
            </div>
            <button onClick={onClose} className="p-2 rounded-full transition-colors" style={{ color: 'var(--text-muted)' }}>
              <X />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
            {loading ? (
              <div className="flex flex-col items-center justify-center py-20 gap-4">
                <div className="w-8 h-8 border-2 border-[#D4AF37]/30 border-t-[#D4AF37] rounded-full animate-spin" />
                <p className="text-sm animate-pulse uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>Retrieving Logs...</p>
              </div>
            ) : history.length === 0 ? (
               <div className="flex flex-col items-center justify-center py-20 uppercase tracking-widest font-bold text-xs gap-4" style={{ color: 'var(--text-muted)' }}>
                 <AlertCircle size={32} />
                 No History Recorded
               </div>
            ) : (
              <div className="grid gap-3">
                {history.map((item, idx) => (
                  <motion.div 
                    key={item.id}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.05 }}
                    className="group flex flex-col gap-3 p-4 rounded-xl transition-all cursor-pointer"
                    style={{ background: 'var(--surface)', border: '1px solid var(--border)', }}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 text-[10px] uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>
                        <Clock size={12} />
                        {item.date}
                      </div>
                      <div className={`px-2 py-0.5 rounded text-[8px] font-bold uppercase tracking-tighter ${
                        item.status === 'verified' ? 'bg-[#D4AF37]/10 text-[#D4AF37] border border-[#D4AF37]/20' : 'bg-[#D4AF37]/10 text-[#D4AF37] border border-[#D4AF37]/20'
                      }`}>
                        {item.status}
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="mt-1">
                        <Search size={14} className="text-white/20 group-hover:text-purple-400 transition-colors" />
                      </div>
                      <p className="text-sm transition-colors line-clamp-2" style={{ color: 'var(--text)' }}>
                        {item.query}
                      </p>
                    </div>
                    
                    {item.metrics && (
                      <div className="flex items-center gap-4 mt-2 pt-3 opacity-40 group-hover:opacity-100 transition-opacity" style={{ borderTop: '1px solid var(--border)' }}>
                        <div className="flex flex-col">
                          <span className="text-[8px] uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>Retrieval</span>
                          <span className="text-[10px] font-mono text-purple-400">{item.metrics.retrieval_lat.toFixed(3)}s</span>
                        </div>
                        <div className="flex flex-col">
                          <span className="text-[8px] uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>E2E Latency</span>
                          <span className="text-[10px] font-mono" style={{ color: 'var(--text-secondary)' }}>{item.metrics.e2e_lat.toFixed(2)}s</span>
                        </div>
                        <div className="flex flex-col">
                          <span className="text-[8px] uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>Words</span>
                          <span className="text-[10px] font-mono" style={{ color: 'var(--text-secondary)' }}>{item.metrics.ans_length}</span>
                        </div>
                      </div>
                    )}
                  </motion.div>
                ))}
              </div>
            )}
          </div>

          <div className="p-4 flex items-center justify-center" style={{ background: 'var(--surface)', borderTop: '1px solid var(--border)' }}>
            <button 
              onClick={() => {
                authFetch(`${API_URL}/api/history`, { method: "DELETE" })
                  .then(() => setHistory([]))
                  .catch(console.error);
              }}
              className="text-[10px] font-bold hover:text-red-400 uppercase tracking-[0.2em] transition-colors" style={{ color: 'var(--text-muted)' }}
            >
              Clear History Log
            </button>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
