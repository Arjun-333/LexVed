"use client";

import React, { useState, useEffect } from "react";
import { X, History, Search, Clock, CheckCircle2, AlertCircle } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

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
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (isOpen) {
      setLoading(true);
      const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:5000";
      fetch(`${API_URL}/api/history`)
        .then((res) => res.json())
        .then((data) => {
          setHistory(data);
          setLoading(false);
        })
        .catch((err) => {
          console.error(err);
          setLoading(false);
        });
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/80 backdrop-blur-md p-4">
        <motion.div 
          initial={{ opacity: 0, scale: 0.9, y: 30 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.9, y: 30 }}
          className="w-full max-w-2xl bg-[#0a0a0a] border border-white/10 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[80vh]"
        >
          <div className="p-6 border-b border-white/10 flex items-center justify-between bg-gradient-to-r from-purple-500/5 to-transparent">
            <div>
              <h2 className="text-xl font-bold text-white flex items-center gap-3">
                <History className="text-purple-400 w-5 h-5" />
                Research History
              </h2>
              <p className="text-white/40 text-xs mt-1">Audit log of institutional queries</p>
            </div>
            <button onClick={onClose} className="p-2 hover:bg-white/5 rounded-full transition-colors">
              <X className="text-white/40 hover:text-white" />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
            {loading ? (
              <div className="flex flex-col items-center justify-center py-20 gap-4">
                <div className="w-8 h-8 border-2 border-purple-500/30 border-t-purple-500 rounded-full animate-spin" />
                <p className="text-white/30 text-sm animate-pulse uppercase tracking-widest">Retrieving Logs...</p>
              </div>
            ) : history.length === 0 ? (
               <div className="flex flex-col items-center justify-center py-20 text-white/20 uppercase tracking-widest font-bold text-xs gap-4">
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
                    className="group flex flex-col gap-3 p-4 rounded-xl border border-white/5 bg-white/[0.02] hover:bg-white/[0.05] hover:border-purple-500/30 transition-all cursor-pointer"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 text-[10px] text-white/40 uppercase tracking-widest">
                        <Clock size={12} />
                        {item.date}
                      </div>
                      <div className={`px-2 py-0.5 rounded text-[8px] font-bold uppercase tracking-tighter ${
                        item.status === 'verified' ? 'bg-green-500/10 text-green-400 border border-green-500/20' : 'bg-yellow-500/10 text-yellow-500 border border-yellow-500/20'
                      }`}>
                        {item.status}
                      </div>
                    </div>
                    <div className="flex items-start gap-3">
                      <div className="mt-1">
                        <Search size={14} className="text-white/20 group-hover:text-purple-400 transition-colors" />
                      </div>
                      <p className="text-sm text-white/80 group-hover:text-white transition-colors line-clamp-2">
                        {item.query}
                      </p>
                    </div>
                    
                    {item.metrics && (
                      <div className="flex items-center gap-4 mt-2 pt-3 border-t border-white/5 opacity-40 group-hover:opacity-100 transition-opacity">
                        <div className="flex flex-col">
                          <span className="text-[8px] uppercase tracking-widest text-white/30">Retrieval</span>
                          <span className="text-[10px] font-mono text-purple-400">{item.metrics.retrieval_lat.toFixed(3)}s</span>
                        </div>
                        <div className="flex flex-col">
                          <span className="text-[8px] uppercase tracking-widest text-white/30">E2E Latency</span>
                          <span className="text-[10px] font-mono text-white/80">{item.metrics.e2e_lat.toFixed(2)}s</span>
                        </div>
                        <div className="flex flex-col">
                          <span className="text-[8px] uppercase tracking-widest text-white/30">Words</span>
                          <span className="text-[10px] font-mono text-white/80">{item.metrics.ans_length}</span>
                        </div>
                      </div>
                    )}
                  </motion.div>
                ))}
              </div>
            )}
          </div>

          <div className="p-4 bg-black/40 border-t border-white/5 flex items-center justify-center">
            <button 
              onClick={() => {
                const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:5000";
                fetch(`${API_URL}/api/history`, { method: "DELETE" })
                  .then(() => setHistory([]))
                  .catch(console.error);
              }}
              className="text-[10px] font-bold text-white/30 hover:text-red-400 uppercase tracking-[0.2em] transition-colors"
            >
              Clear History Log
            </button>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
