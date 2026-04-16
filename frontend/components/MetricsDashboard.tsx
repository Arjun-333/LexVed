"use client";

import React, { useState, useEffect, useRef } from "react";
import { X, ShieldCheck, Zap, Activity, HardDrive, Download, Loader2, FileSpreadsheet, Play, Database, ChevronDown } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import jsPDF from "jspdf";
import EmbeddingOmnitrix from "./EmbeddingOmnitrix";

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
  const [selectedModel, setSelectedModel] = useState("multi-qa-mpnet-base-cos-v1");
  const [selectedDb, setSelectedDb] = useState("qdrant");
  const [isStartingEval, setIsStartingEval] = useState(false);
  const [isDbDropdownOpen, setIsDbDropdownOpen] = useState(false);
  const pollRef = useRef<NodeJS.Timeout | null>(null);

  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:5000";

  const fetchMetrics = () => {
    fetch(`${API_URL}/api/metrics`)
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
      
      // Fetch current settings
      fetch(`${API_URL}/api/settings/embedding_model`)
        .then(res => res.json())
        .then(data => setSelectedModel(data.model))
        .catch(console.error);

      fetch(`${API_URL}/api/settings/vector_db`)
        .then(res => res.json())
        .then(data => setSelectedDb(data.db))
        .catch(console.error);
        
      pollRef.current = setInterval(fetchMetrics, 3000);
    }
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [isOpen]);

  const handleModelChange = async (modelId: string) => {
    setSelectedModel(modelId);
    try {
      await fetch(`${API_URL}/api/settings/embedding_model`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: modelId }),
      });
    } catch (err) {
      console.error("Failed to update model setting", err);
    }
  };

  const handleDbChange = async (dbId: string) => {
    setSelectedDb(dbId);
    setIsDbDropdownOpen(false);
    try {
      await fetch(`${API_URL}/api/settings/vector_db`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ db: dbId }),
      });
    } catch (err) {
      console.error("Failed to update DB setting", err);
    }
  };

  const handleStartEvaluation = async () => {
    setIsStartingEval(true);
    try {
      const res = await fetch(`${API_URL}/api/workflow/evaluate`, { method: "POST" });
      const data = await res.json();
      if (data.status === "processing") {
        fetchMetrics();
      }
    } catch (err) {
      console.error("Failed to start evaluation", err);
    } finally {
      setIsStartingEval(false);
    }
  };

  const isProcessing = metrics?.status === "processing";

  const getVal = (key: string): number | null => {
    const v = metrics?.summary?.[key];
    return (v !== null && v !== undefined) ? v : null;
  };

  const rows: MetricRow[] = metrics?.summary ? [
    { id: "M1",  category: "Retrieval",   label: "Embedding Latency",     value: getVal("M1"),   decimals: 4, unit: "s" },
    { id: "M2",  category: "Retrieval",   label: "Index Point Count",     value: getVal("M2"),   decimals: 0, unit: "vec" },
    { id: "M3",  category: "Retrieval",   label: "Retrieval Latency",     value: getVal("M3"),   decimals: 4, unit: "s" },
    { id: "M4",  category: "Retrieval",   label: "Cosine Similarity",     value: getVal("M4"),   decimals: 4, unit: "score" },
    { id: "M5",  category: "Retrieval",   label: "Recall@K",              value: getVal("M5"),   decimals: 4, unit: "score" },
    { id: "M6",  category: "Quality",     label: "ROUGE-1 Score",         value: getVal("M6"),   decimals: 4, unit: "score" },
    { id: "M7",  category: "Quality",     label: "ROUGE-2 Score",         value: getVal("M7"),   decimals: 4, unit: "score" },
    { id: "M8",  category: "Quality",     label: "ROUGE-L Score",         value: getVal("M8"),   decimals: 4, unit: "score" },
    { id: "M9",  category: "Quality",     label: "METEOR Score",          value: getVal("M9"),   decimals: 4, unit: "score" },
    { id: "M10", category: "Quality",     label: "BLEU Score",            value: getVal("M10"),  decimals: 4, unit: "score" },
    { id: "M11", category: "Quality",     label: "SemScore (Approx)",     value: getVal("M11"),  decimals: 4, unit: "score" },
    { id: "M12", category: "Quality",     label: "BERTScore (F1)",        value: getVal("M12"),  decimals: 4, unit: "score" },
    { id: "M13", category: "Safety",      label: "Hallucination Rate",    value: getVal("M13"),  decimals: 0, unit: "%" },
    { id: "M14", category: "Quality",     label: "Faithfulness",          value: getVal("M14"),  decimals: 0, unit: "%" },
    { id: "M15", category: "Quality",     label: "Factual Consistency",   value: getVal("M15"),  decimals: 0, unit: "%" },
    { id: "M16", category: "Efficiency",  label: "End-to-End Latency",    value: getVal("M16"),  decimals: 2, unit: "s" },
    { id: "M17", category: "Efficiency",  label: "Token Gen Latency",     value: getVal("M17"),  decimals: 4, unit: "s" },
    { id: "M18", category: "Efficiency",  label: "Cost Per Query",        value: getVal("M18"),  decimals: 4, unit: "$" },
    { id: "M19", category: "Efficiency",  label: "RAM Utilization",       value: getVal("M19"),  decimals: 2, unit: "MB" },
    { id: "M20", category: "Legal",       label: "Citation Accuracy",     value: getVal("M20"),  decimals: 0, unit: "%" },
    { id: "M21", category: "Legal",       label: "Term Precision",        value: getVal("M21"),  decimals: 0, unit: "%" },
    { id: "M22", category: "Legal",       label: "Precedent Match",       value: getVal("M22"),  decimals: 0, unit: "%" },
    { id: "M23", category: "Legal",       label: "Regulatory Align.",     value: getVal("M23"),  decimals: 0, unit: "%" },
    { id: "M24", category: "Legal",       label: "Jurisdictional Comp.",  value: getVal("M24"),  decimals: 0, unit: "%" },
  ] : [];

  const handleExportPDF = () => {
    if (!metrics || !metrics.summary) return;
    const doc = new jsPDF();
    const timestamp = metrics.timestamp || new Date().toLocaleString();
    const sys = metrics.system_info || { 
      vector_db: selectedDb, 
      model: "Llama 3 8B (Local)", 
      embedding: selectedModel
    };

    doc.setFontSize(20);
    doc.setTextColor(180, 150, 40);
    doc.text("LexVed Institutional Audit Report", 14, 20);
    doc.setFontSize(9);
    doc.setTextColor(120);
    doc.text(`Mission-Critical Metrics (M1-M24) | ${timestamp}`, 14, 27);
    doc.text(`System: ${sys.model} | Vector DB: ${sys.vector_db.toUpperCase()}`, 14, 32);

    const colX = [14, 30, 62, 120, 165];
    const headers = ["ID", "Category", "Metric Label", "Value", "Unit"];
    let y = 42;
    doc.setFillColor(20, 20, 20);
    doc.rect(12, y - 5, 186, 8, "F");
    doc.setFontSize(8);
    doc.setTextColor(200, 170, 50);
    headers.forEach((h, i) => doc.text(h, colX[i], y));
    y += 6;

    rows.forEach((row, idx) => {
      if (y > 275) { doc.addPage(); y = 20; }
      if (idx % 2 === 0) {
        doc.setFillColor(245, 245, 245);
        doc.rect(12, y - 4, 186, 7, "F");
      }
      doc.setTextColor(60);
      const val = row.value !== null ? Number(row.value).toFixed(row.decimals) : "Pending";
      doc.text(String(row.id), colX[0], y);
      doc.text(String(row.category), colX[1], y);
      doc.text(String(row.label), colX[2], y);
      doc.text(String(val), colX[3], y);
      doc.text(String(row.unit), colX[4], y);
      y += 7;
    });

    const dateStr = new Date().toISOString().split('T')[0];
    doc.save(`LexVed_Audit_Report_${dateStr}.pdf`);
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
        <motion.div 
          initial={{ opacity: 0, scale: 0.95, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 20 }}
          className="w-full max-w-5xl bg-[#0a0a0a] border border-[#d4af37]/30 rounded-2xl shadow-[0_0_50px_rgba(212,175,55,0.1)] overflow-hidden flex flex-col max-h-[90vh]"
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
                <Loader2 className="w-12 h-12 text-[#d4af37] animate-spin" />
                <p className="text-[#d4af37] font-medium pulse">Analyzing Intelligence Node...</p>
              </div>
            ) : (
              <div className="flex flex-col lg:flex-row gap-8">
                {/* Left: Controls */}
                <div className="lg:w-1/3 flex flex-col gap-6">
                   {/* Custom DB Selector */}
                   <div className="p-5 bg-white/5 border border-white/10 rounded-2xl relative">
                     <div className="flex items-center gap-2 text-[#d4af37] mb-3">
                       <Database className="w-4 h-4" />
                       <span className="text-[10px] uppercase font-bold tracking-[0.2em]">Vector Infrastructure</span>
                     </div>
                     
                     <div className="relative">
                       <button 
                         onClick={() => setIsDbDropdownOpen(!isDbDropdownOpen)}
                         className="w-full bg-black border border-[#d4af37]/30 text-white rounded-lg p-3 text-sm flex items-center justify-between hover:border-[#d4af37] transition-all"
                       >
                         <span className="font-medium text-white">
                           {selectedDb === "qdrant" ? "QDRANT (Self-Hosted)" : "PINECONE (Serverless)"}
                         </span>
                         <ChevronDown className={`w-4 h-4 text-[#d4af37] transition-transform ${isDbDropdownOpen ? 'rotate-180' : ''}`} />
                       </button>

                       <AnimatePresence>
                         {isDbDropdownOpen && (
                           <motion.div 
                             initial={{ opacity: 0, y: -10 }}
                             animate={{ opacity: 1, y: 0 }}
                             exit={{ opacity: 0, y: -10 }}
                             className="absolute top-full left-0 right-0 mt-2 bg-[#1a1a1a] border border-[#d4af37]/30 rounded-lg overflow-hidden z-20 shadow-2xl"
                           >
                             {[
                               { id: "qdrant", label: "QDRANT (Self-Hosted)" },
                               { id: "pinecone", label: "PINECONE (Serverless)" }
                             ].map((opt) => (
                               <button
                                 key={opt.id}
                                 onClick={() => handleDbChange(opt.id)}
                                 className={`w-full text-left p-3 text-sm transition-colors ${selectedDb === opt.id ? 'bg-[#d4af37] text-black font-bold' : 'text-white hover:bg-white/5'}`}
                               >
                                 {opt.label}
                               </button>
                             ))}
                           </motion.div>
                         )}
                       </AnimatePresence>
                     </div>
                   </div>

                   <EmbeddingOmnitrix 
                     selectedModel={selectedModel} 
                     onSelect={handleModelChange} 
                   />
                   
                   <motion.button
                     whileHover={{ scale: 1.02 }}
                     whileTap={{ scale: 0.98 }}
                     onClick={handleStartEvaluation}
                     disabled={isProcessing || isStartingEval}
                     className={`w-full py-4 rounded-xl flex items-center justify-center gap-3 font-bold uppercase tracking-widest text-xs transition-all
                       ${isProcessing || isStartingEval 
                         ? 'bg-white/5 text-white/20 border border-white/5' 
                         : 'bg-[var(--accent)] text-black border border-[var(--accent-glow)] shadow-[0_0_20px_rgba(212,175,55,0.3)] hover:shadow-[0_0_30px_rgba(212,175,55,0.5)]'}`}
                   >
                     {isStartingEval ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4 fill-current" />}
                     {isProcessing ? "Benchmarking..." : "Start Evaluation"}
                   </motion.button>

                   <div className="flex gap-2">
                      <button onClick={handleExportCSV} className="flex-1 py-3 bg-green-500/10 border border-green-500/30 text-green-400 rounded-lg text-[10px] font-bold hover:bg-green-500 hover:text-black transition-all">
                        EXCEL
                      </button>
                      <button onClick={handleExportPDF} className="flex-1 py-3 bg-[#d4af37]/10 border border-[#d4af37]/30 text-[#d4af37] rounded-lg text-[10px] font-bold hover:bg-[#d4af37] hover:text-black transition-all">
                        EXPORT PDF
                      </button>
                   </div>
                </div>

                {/* Right: Metrics */}
                <div className="lg:w-2/3 space-y-6">
                  <div className="overflow-hidden border border-white/10 rounded-xl bg-white/[0.02]">
                    <div className="px-6 py-4 border-b border-white/10 bg-white/5 flex justify-between items-center">
                      <span className="text-[10px] uppercase font-bold tracking-widest text-white/40">Audit Metrics Report</span>
                      <span className="text-[10px] text-[#d4af37] font-mono">{metrics?.system_info?.vector_db?.toUpperCase() || "NODE"} ACTIVE</span>
                    </div>
                    <table className="w-full text-left">
                      <thead className="text-white/40 text-[10px] font-bold uppercase tracking-wider">
                        <tr>
                          <th className="px-6 py-3">ID</th>
                          <th className="px-6 py-3">Metric Label</th>
                          <th className="px-6 py-3">Value</th>
                          <th className="px-3 py-3">Unit</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-white/5">
                        {rows.map((row) => (
                          <tr key={row.id} className="hover:bg-white/[0.02] transition-colors group">
                            <td className="px-6 py-3 font-mono text-[#d4af37] text-xs">{row.id}</td>
                            <td className="px-6 py-3 text-white/80 text-xs font-medium">{row.label}</td>
                            <td className="px-6 py-3 font-mono text-xs">
                              {row.value !== null ? (
                                <span className="text-green-400">{row.value.toFixed(row.decimals)}</span>
                              ) : (
                                <span className="text-white/10">PENDING</span>
                              )}
                            </td>
                            <td className="px-3 py-3 text-white/30 text-[10px]">{row.unit}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <div className="grid grid-cols-3 gap-4">
                    {[
                      { icon: Zap, label: "E2E Latency", val: metrics?.summary?.M16, suffix: "s" },
                      { icon: Activity, label: "Faithfulness", val: metrics?.summary?.M14, suffix: "%" },
                      { icon: HardDrive, label: "Vector Count", val: metrics?.summary?.M2, suffix: "" }
                    ].map((card, i) => (
                      <div key={i} className="p-4 bg-white/5 border border-white/10 rounded-xl relative overflow-hidden group">
                        <div className="absolute top-0 right-0 p-2 opacity-10 group-hover:opacity-30 transition-opacity">
                          <card.icon className="w-8 h-8 text-[#d4af37]" />
                        </div>
                        <div className="text-white/40 text-[8px] uppercase font-bold tracking-widest mb-1">{card.label}</div>
                        <div className="text-white text-lg font-bold">
                          {card.val ? `${Number(card.val).toFixed(i === 2 ? 0 : 2)}${card.suffix}` : "—"}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );

  function handleExportCSV() {
    if (!metrics || !metrics.summary) return;
    const header = "Metric ID,Category,Metric Label,Evaluated Value,Unit";
    const csvRows = rows.map(r => `${r.id},${r.category},"${r.label}",${r.value?.toFixed(r.decimals) || "N/A"},${r.unit}`);
    const csvContent = `${header}\n${csvRows.join("\n")}`;
    const csvBlob = new Blob([csvContent], { type: "text/csv" });
    const csvUrl = URL.createObjectURL(csvBlob);
    const csvLink = document.createElement("a");
    csvLink.href = csvUrl;
    csvLink.download = `audit_${selectedDb}_${new Date().toISOString().split('T')[0]}.csv`;
    csvLink.click();
  }
}
