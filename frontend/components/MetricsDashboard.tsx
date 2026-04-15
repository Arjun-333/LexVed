"use client";

import React, { useState, useEffect, useRef } from "react";
import { X, ShieldCheck, Zap, Activity, HardDrive, Download, Loader2, FileSpreadsheet } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import jsPDF from "jspdf";

interface MetricRow {
  id: string;
  category: string;
  value: number | null;
  unit: string;
  label: string;
  decimals: number;
}

export default function MetricsDashboard({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const [metrics, setMetrics] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const pollRef = useRef<NodeJS.Timeout | null>(null);

  const fetchMetrics = () => {
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
  };

  useEffect(() => {
    if (isOpen) {
      setLoading(true);
      fetchMetrics();
      pollRef.current = setInterval(fetchMetrics, 3000);
    }
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [isOpen]);

  const isProcessing = metrics?.status === "processing";

  const fmt = (v: any, d: number = 4): string | null => {
    if (v === null || v === undefined) return null;
    if (typeof v === 'number') return v.toFixed(d);
    return String(v);
  };

  const getVal = (key: string): number | null => {
    const v = metrics?.summary?.[key];
    if (v === null || v === undefined) return null;
    return v;
  };

  const rows: MetricRow[] = metrics?.summary ? [
    { id: "M1",  category: "Retrieval",   label: "Embedding Latency",     value: getVal("M1"),   decimals: 4, unit: "sec" },
    { id: "M2",  category: "Retrieval",   label: "Index Point Count",     value: getVal("M2"),   decimals: 0, unit: "vectors" },
    { id: "M3",  category: "Retrieval",   label: "Retrieval Latency",     value: getVal("M3"),   decimals: 4, unit: "sec" },
    { id: "M4",  category: "Retrieval",   label: "Cosine Similarity",     value: getVal("M4"),   decimals: 4, unit: "score" },
    { id: "M5",  category: "Retrieval",   label: "Recall@K",              value: getVal("M5"),   decimals: 4, unit: "score" },
    { id: "M6",  category: "Quality",     label: "ROUGE-1 Score",         value: getVal("M6"),   decimals: 4, unit: "score" },
    { id: "M7",  category: "Quality",     label: "ROUGE-2 Score",         value: getVal("M7"),   decimals: 4, unit: "score" },
    { id: "M8",  category: "Quality",     label: "ROUGE-L Score",         value: getVal("M8"),   decimals: 4, unit: "score" },
    { id: "M9",  category: "Quality",     label: "METEOR Score",          value: getVal("M9"),   decimals: 4, unit: "score" },
    { id: "M10", category: "Quality",     label: "BLEU Score",            value: getVal("M10"),  decimals: 4, unit: "score" },
    { id: "M11", category: "Quality",     label: "Answer Length",         value: getVal("M11"),  decimals: 0, unit: "tokens" },
    { id: "M12", category: "Quality",     label: "BERTScore (F1)",        value: getVal("M12"),  decimals: 4, unit: "score" },
    { id: "M13", category: "Quality",     label: "Factual Consistency",   value: getVal("M13"),  decimals: 0, unit: "%" },
    { id: "M14", category: "Quality",     label: "Faithfulness",          value: getVal("M14"),  decimals: 0, unit: "%" },
    { id: "M15", category: "Quality",     label: "Semantic Similarity",   value: getVal("M15"),  decimals: 4, unit: "score" },
    { id: "M16", category: "Efficiency",  label: "End-to-End Latency",    value: getVal("M16"),  decimals: 2, unit: "sec" },
    { id: "M17", category: "Efficiency",  label: "Throughput",            value: getVal("M17"),  decimals: 2, unit: "q/min" },
    { id: "M18", category: "Efficiency",  label: "CPU Utilization",       value: getVal("M18"),  decimals: 1, unit: "%" },
    { id: "M19", category: "Efficiency",  label: "RAM Delta",             value: getVal("M19"),  decimals: 2, unit: "MB" },
    { id: "M20", category: "Legal",       label: "Citation Accuracy",     value: getVal("M20"),  decimals: 0, unit: "%" },
    { id: "M21", category: "Legal",       label: "Term Precision",        value: getVal("M21"),  decimals: 0, unit: "%" },
    { id: "M22", category: "Legal",       label: "Precedent Match",       value: getVal("M22"),  decimals: 0, unit: "%" },
    { id: "M23", category: "Legal",       label: "Hallucination Rate",    value: getVal("M23"),  decimals: 2, unit: "%" },
    { id: "M24", category: "Legal",       label: "Bias Detection",        value: getVal("M24"),  decimals: 0, unit: "score" },
  ] : [];

  const handleExportPDF = () => {
    if (!metrics || !metrics.summary) return;
    const doc = new jsPDF();
    const timestamp = metrics.timestamp || new Date().toLocaleString();

    // Header
    doc.setFontSize(20);
    doc.setTextColor(180, 150, 40);
    doc.text("LexVed Institutional Audit Report", 14, 20);
    doc.setFontSize(9);
    doc.setTextColor(120);
    doc.text(`Mission-Critical Metrics (M1-M24) | ${timestamp}`, 14, 27);
    doc.text("System: Llama 3 8B (Local) | Vector DB: Qdrant | Encryption: AES-256", 14, 32);

    // Table header
    const colX = [14, 30, 62, 120, 162];
    const headers = ["ID", "Category", "Metric Label", "Value", "Unit"];
    let y = 42;

    doc.setFillColor(20, 20, 20);
    doc.rect(12, y - 5, 186, 8, "F");
    doc.setFontSize(8);
    doc.setTextColor(200, 170, 50);
    headers.forEach((h, i) => doc.text(h, colX[i], y));
    y += 6;

    // Table rows
    doc.setFontSize(8);
    rows.forEach((row, idx) => {
      if (y > 275) { doc.addPage(); y = 20; }
      if (idx % 2 === 0) {
        doc.setFillColor(245, 245, 245);
        doc.rect(12, y - 4, 186, 7, "F");
      }
      doc.setTextColor(60);
      const val = row.value !== null ? row.value.toFixed(row.decimals) : "Pending";
      doc.text(row.id, colX[0], y);
      doc.text(row.category, colX[1], y);
      doc.text(row.label, colX[2], y);
      doc.text(val, colX[3], y);
      doc.text(row.unit, colX[4], y);
      y += 7;
    });

    // Footer
    y += 8;
    doc.setFontSize(7);
    doc.setTextColor(150);
    doc.text("LexVed Confidential Audit Protocol 2.0 | Unauthorized duplication prohibited.", 14, y);

    doc.save(`LexVed_Audit_${Date.now()}.pdf`);
  };

  const handleExportCSV = () => {
    if (!metrics || !metrics.summary) return;
    const timestamp = metrics.timestamp || new Date().toLocaleString();
    const header = "Metric ID,Category,Metric Label,Evaluated Value,Unit";
    const csvRows = rows.map(r => {
      const val = r.value !== null ? r.value.toFixed(r.decimals) : "Pending";
      return `${r.id},${r.category},"${r.label}",${val},${r.unit}`;
    });
    const csv = `LexVed Performance Audit - ${timestamp}\n${header}\n${csvRows.join("\n")}`;
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `LexVed_Audit_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (!isOpen) return null;

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
            <div className="flex items-center gap-4">
              {!loading && metrics?.summary && (
                <>
                  <button 
                    onClick={handleExportCSV}
                    className="flex items-center gap-2 px-4 py-2 bg-green-500/10 border border-green-500/30 text-green-400 rounded-lg text-xs font-bold hover:bg-green-500 hover:text-black transition-all"
                  >
                    <FileSpreadsheet className="w-4 h-4" />
                    EXCEL
                  </button>
                  <button 
                    onClick={handleExportPDF}
                    className="flex items-center gap-2 px-4 py-2 bg-[#d4af37]/10 border border-[#d4af37]/30 text-[#d4af37] rounded-lg text-xs font-bold hover:bg-[#d4af37] hover:text-black transition-all"
                  >
                    <Download className="w-4 h-4" />
                    EXPORT PDF
                  </button>
                </>
              )}
              <button onClick={onClose} className="p-2 hover:bg-white/5 rounded-full transition-colors">
                <X className="text-white/50 hover:text-white" />
              </button>
            </div>
          </div>

          {/* Processing Banner */}
          {isProcessing && (
            <div className="px-6 py-3 bg-[#d4af37]/10 border-b border-[#d4af37]/20 flex items-center gap-3">
              <Loader2 className="w-4 h-4 text-[#d4af37] animate-spin" />
              <span className="text-[#d4af37] text-xs font-bold uppercase tracking-wider">
                Live Audit In Progress — {metrics.progress || "Processing..."}
              </span>
            </div>
          )}

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
                      {rows.map((row) => {
                        const hasValue = row.value !== null && row.value !== undefined;
                        return (
                          <tr key={row.id} className="hover:bg-white/[0.02] transition-colors group">
                            <td className="px-6 py-4 font-mono text-[#d4af37] text-sm">{row.id}</td>
                            <td className="px-6 py-4">
                              <span className="px-2 py-1 rounded-md bg-white/5 text-white/50 text-[10px] font-bold">
                                {row.category}
                              </span>
                            </td>
                            <td className="px-6 py-4 text-white font-medium">{row.label}</td>
                            <td className="px-6 py-4 font-mono">
                              {hasValue ? (
                                <motion.span 
                                  initial={{ opacity: 0, x: -10 }} 
                                  animate={{ opacity: 1, x: 0 }} 
                                  className="text-green-400"
                                >
                                  {row.value!.toFixed(row.decimals)}
                                </motion.span>
                              ) : (
                                <span className="flex items-center gap-2 text-white/20">
                                  <Loader2 className="w-3 h-3 animate-spin" />
                                  <span className="text-xs">Evaluating...</span>
                                </span>
                              )}
                            </td>
                            <td className="px-6 py-4 text-white/30 text-xs">{row.unit}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-8">
                  <div className="p-4 bg-white/5 border border-white/10 rounded-xl">
                    <Zap className="text-[#d4af37] w-5 h-5 mb-2" />
                    <div className="text-white/40 text-xs uppercase font-bold tracking-widest">Avg E2E Latency</div>
                    <div className="text-white text-2xl font-bold">{metrics.summary.M16 ? metrics.summary.M16.toFixed(2) + "s" : "—"}</div>
                  </div>
                  <div className="p-4 bg-white/5 border border-white/10 rounded-xl">
                    <Activity className="text-[#d4af37] w-5 h-5 mb-2" />
                    <div className="text-white/40 text-xs uppercase font-bold tracking-widest">Faithfulness</div>
                    <div className="text-white text-2xl font-bold">{metrics.summary.M14 ? metrics.summary.M14 + "%" : "—"}</div>
                  </div>
                  <div className="p-4 bg-white/5 border border-white/10 rounded-xl">
                    <HardDrive className="text-[#d4af37] w-5 h-5 mb-2" />
                    <div className="text-white/40 text-xs uppercase font-bold tracking-widest">Total Vectors</div>
                    <div className="text-white text-2xl font-bold">{metrics.summary.M2 || "—"}</div>
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
               <div className={`w-1.5 h-1.5 rounded-full ${isProcessing ? 'bg-yellow-500' : 'bg-green-500'} animate-pulse`}></div>
               <span className={`text-[10px] ${isProcessing ? 'text-yellow-500/80' : 'text-green-500/80'} font-bold uppercase py-0.5`}>
                 {isProcessing ? "Processing" : "Secure Node"}
               </span>
            </div>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
