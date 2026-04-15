"use client";

import React, { useState, useEffect } from "react";
import { X, ShieldCheck, Zap, Activity, HardDrive } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface MetricRow {
  id: string;
  category: string;
  value: string | number;
  unit: string;
  label: string;
}

export default function MetricsDashboard({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const [metrics, setMetrics] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (isOpen) {
      fetch("http://localhost:5000/api/metrics")
        .then((res) => res.json())
        .then((data) => {
          setMetrics(data);
          setLoading(false);
        })
        .catch((err) => {
          console.error(err);
          setLoading(false);
        });
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const rows: MetricRow[] = metrics?.summary ? [
    { id: "M1", category: "Retrieval", label: "Embedding Latency", value: metrics.summary.M1?.toFixed(4) || "0.00", unit: "sec" },
    { id: "M2", category: "Retrieval", label: "Index Point Count", value: metrics.summary.M2 || "0", unit: "vectors" },
    { id: "M3", category: "Retrieval", label: "Retrieval Latency", value: metrics.summary.M3?.toFixed(4) || "0.00", unit: "sec" },
    { id: "M4", category: "Retrieval", label: "Cosine Similarity", value: metrics.summary.M4?.toFixed(4) || "0.00", unit: "score" },
    { id: "M6", category: "Quality", label: "ROUGE-1 Score", value: metrics.summary.M6?.toFixed(4) || "0.00", unit: "score" },
    { id: "M10", category: "Quality", label: "BLEU Score", value: metrics.summary.M10?.toFixed(4) || "0.00", unit: "score" },
    { id: "M12", category: "Quality", label: "BERTScore (F1)", value: metrics.summary.M12?.toFixed(4) || "0.00", unit: "score" },
    { id: "M14", category: "Quality", label: "Faithfulness (%)", value: metrics.summary.M14 || "0", unit: "%" },
    { id: "M16", category: "Efficiency", label: "End-to-End Latency", value: metrics.summary.M16?.toFixed(2) || "0.00", unit: "sec" },
    { id: "M20", category: "Legal", label: "Citation Accuracy", value: metrics.summary.M20 || "0", unit: "%" },
    { id: "M21", category: "Legal", label: "Term Precision", value: metrics.summary.M21 || "0", unit: "%" },
  ] : [];

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
        <motion.div 
          initial={{ opacity: 0, scale: 0.95, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 20 }}
          className="w-full max-w-4xl bg-[#0a0a0a] border border-[#d4af37]/30 rounded-2xl shadow-[0_0_50px_rgba(212,175,55,0.1)] overflow-hidden flex flex-col max-h-[90vh]"
        >
          {/* Header */}
          <div className="p-6 border-b border-white/10 flex items-center justify-between bg-gradient-to-r from-[#d4af37]/5 to-transparent">
            <div>
              <h2 className="text-2xl font-bold text-white flex items-center gap-3">
                <ShieldCheck className="text-[#d4af37] w-6 h-6" />
                LexVed Performance Audit
              </h2>
              <p className="text-white/50 text-sm mt-1">Institutional RAG Benchmarking — Mission-Critical (M1-M24)</p>
            </div>
            <button onClick={onClose} className="p-2 hover:bg-white/5 rounded-full transition-colors">
              <X className="text-white/50 hover:text-white" />
            </button>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto p-6">
            {loading ? (
              <div className="flex flex-col items-center justify-center py-20 gap-4">
                <div className="w-12 h-12 border-4 border-[#d4af37] border-t-transparent rounded-full animate-spin"></div>
                <p className="text-[#d4af37] font-medium pulse">Analyzing Intelligence Node...</p>
              </div>
            ) : metrics?.status === "error" ? (
              <div className="text-center py-20">
                <Activity className="w-16 h-16 text-red-500/50 mx-auto mb-4" />
                <h3 className="text-xl text-white font-medium">Audit Results Delayed</h3>
                <p className="text-white/40 mt-2 max-w-sm mx-auto">{metrics.message}</p>
                <div className="mt-8 p-4 bg-white/5 border border-white/10 rounded-lg text-left inline-block">
                  <code className="text-[#d4af37] text-sm font-mono">cd backend && ./venv/bin/python3 run_metrics.py</code>
                </div>
              </div>
            ) : (
              <div className="space-y-6">
                {/* Summary Table */}
                <div className="overflow-hidden border border-white/10 rounded-xl">
                  <table className="w-full text-left">
                    <thead className="bg-white/5 text-white/40 text-xs font-bold uppercase tracking-wider">
                      <tr>
                        <th className="px-6 py-4">Metric ID</th>
                        <th className="px-6 py-4">Target Area</th>
                        <th className="px-6 py-4">Metric Label</th>
                        <th className="px-6 py-4">Evaluated Value</th>
                        <th className="px-6 py-4">Unit</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5">
                      {rows.map((row) => (
                        <tr key={row.id} className="hover:bg-white/[0.02] transition-colors group">
                          <td className="px-6 py-4 font-mono text-[#d4af37] text-sm">{row.id}</td>
                          <td className="px-6 py-4">
                            <span className="px-2 py-1 rounded-md bg-white/5 text-white/50 text-[10px] font-bold">
                              {row.category}
                            </span>
                          </td>
                          <td className="px-6 py-4 text-white font-medium">{row.label}</td>
                          <td className="px-6 py-4 text-white font-mono">{row.value}</td>
                          <td className="px-6 py-4 text-white/30 text-xs">{row.unit}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-8">
                  <div className="p-4 bg-white/5 border border-white/10 rounded-xl">
                    <Zap className="text-[#d4af37] w-5 h-5 mb-2" />
                    <div className="text-white/40 text-xs uppercase font-bold tracking-widest">Avg E2E Latency</div>
                    <div className="text-white text-2xl font-bold">{metrics.summary.M16?.toFixed(2)}s</div>
                  </div>
                  <div className="p-4 bg-white/5 border border-white/10 rounded-xl">
                    <Activity className="text-[#d4af37] w-5 h-5 mb-2" />
                    <div className="text-white/40 text-xs uppercase font-bold tracking-widest">Faithfulness</div>
                    <div className="text-white text-2xl font-bold">{metrics.summary.M14}%</div>
                  </div>
                  <div className="p-4 bg-white/5 border border-white/10 rounded-xl">
                    <HardDrive className="text-[#d4af37] w-5 h-5 mb-2" />
                    <div className="text-white/40 text-xs uppercase font-bold tracking-widest">Total Vectors</div>
                    <div className="text-white text-2xl font-bold">{metrics.summary.M2}</div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="p-4 bg-white/5 border-t border-white/10 flex items-center justify-between">
            <p className="text-[10px] text-white/30 uppercase tracking-[0.2em]">
              Authorized Internal Audit Report • Generated: {metrics?.timestamp || "Pending"}
            </p>
            <div className="flex gap-2">
               <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"></div>
               <span className="text-[10px] text-green-500/80 font-bold uppercase py-0.5">Secure Node</span>
            </div>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
