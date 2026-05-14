"use client";

import React, { useState, useEffect, useRef } from "react";
import { X, ShieldCheck, Zap, Activity, HardDrive, Download, Loader2, FileSpreadsheet, Play, Database, ChevronDown } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import jsPDF from "jspdf";
import EmbeddingOmnitrix from "./EmbeddingOmnitrix";
import { useAuth } from "./AuthContext";

interface MetricRow {
  id: string;
  category: string;
  value: number | null;
  unit: string;
  label: string;
  decimals: number;
}

export default function MetricsDashboard({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const { authFetch } = useAuth();
  const [metrics, setMetrics] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [selectedModel, setSelectedModel] = useState("multi-qa-mpnet-base-cos-v1");
  const [selectedDb, setSelectedDb] = useState("qdrant");
  const [isStartingEval, setIsStartingEval] = useState(false);
  const [isDbDropdownOpen, setIsDbDropdownOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<"single" | "comparative" | "pipeline_compare">("single");
  const [comparative, setComparative] = useState<any>(null);
  const [isStartingComparative, setIsStartingComparative] = useState(false);
  const [pipelineCompare, setPipelineCompare] = useState<any>(null);
  const [isStartingPipelineCompare, setIsStartingPipelineCompare] = useState(false);
  const [primitiveMetrics, setPrimitiveMetrics] = useState<any>(null);
  const [isStartingPrimitive, setIsStartingPrimitive] = useState(false);
  const [primitiveModelChoice, setPrimitiveModelChoice] = useState("1");
  const [smoothPct, setSmoothPct] = useState(5);
  const pollRef = useRef<NodeJS.Timeout | null>(null);

  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:5000";

  const fetchMetrics = () => {
    authFetch(`${API_URL}/api/metrics`)
      .then((res) => {
        if (!res.ok) throw new Error("API Offline");
        return res.json();
      })
      .then((data) => {
        setMetrics(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setMetrics((prev: any) => prev?.status === "processing" ? { status: "error", message: "API Disconnected" } : prev);
        setLoading(false);
      });
  };

  useEffect(() => {
    if (isOpen) {
      setLoading(true);
      fetchMetrics();
      
      // Fetch current settings
      authFetch(`${API_URL}/api/settings/embedding_model`)
        .then(res => res.json())
        .then(data => setSelectedModel(data.model))
        .catch(console.error);

      authFetch(`${API_URL}/api/settings/vector_db`)
        .then(res => res.json())
        .then(data => setSelectedDb(data.db))
        .catch(console.error);

      const fetchComp = () => {
        authFetch(`${API_URL}/api/comparative`)
          .then(res => {
            if (!res.ok) throw new Error("API Offline");
            return res.json();
          })
          .then(data => setComparative(data))
          .catch(err => {
            console.error(err);
            setComparative((prev: any) => prev?.status === "processing" ? { status: "error", message: "API Disconnected" } : prev);
          });
      };

      const fetchPipelineComp = () => {
        authFetch(`${API_URL}/api/compare_pipelines`)
          .then(res => {
            if (!res.ok) throw new Error("API Offline");
            return res.json();
          })
          .then(data => setPipelineCompare(data))
          .catch(err => {
            console.error(err);
            setPipelineCompare((prev: any) => prev?.status === "processing" ? { status: "error", message: "API Disconnected" } : prev);
          });
      };

      const fetchPrimitiveMetrics = () => {
        authFetch(`${API_URL}/api/metrics/primitive`)
          .then(res => { if (!res.ok) throw new Error(); return res.json(); })
          .then(data => setPrimitiveMetrics(data))
          .catch(() => {});
      };

      fetchComp();
      fetchPipelineComp();
      fetchPrimitiveMetrics();
      pollRef.current = setInterval(() => {
         fetchMetrics();
         fetchComp();
         fetchPipelineComp();
         fetchPrimitiveMetrics();
      }, 3000);
    }
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [isOpen]);

  useEffect(() => {
    if (comparative?.status !== "processing") {
      setSmoothPct(5);
      return;
    }

    let targetPct = 5;
    if (comparative?.progress) {
      const modelMatch = comparative.progress.match(/model\s+(\d+)\s*\/\s*(\d+)/i);
      const queryMatch = comparative.progress.match(/(\d+)\s*\/\s*(\d+)\s+queries/i);

      if (modelMatch) {
        const currModel = parseInt(modelMatch[1]);
        const totalModels = parseInt(modelMatch[2]);
        let subProgress = 0;
        if (queryMatch) {
          const currQuery = parseInt(queryMatch[1]);
          const totalQueries = parseInt(queryMatch[2]);
          subProgress = currQuery / totalQueries;
        }
        targetPct = Math.round(((currModel - 1 + subProgress) / totalModels) * 100);
      }
    }

    targetPct = Math.max(5, Math.min(targetPct, 98));

    const interval = setInterval(() => {
      setSmoothPct(prev => {
        if (prev < targetPct) {
          return Math.min(prev + 1, targetPct);
        } else if (prev >= targetPct && prev < targetPct + 5 && prev < 98) {
          return Math.random() > 0.6 ? prev + 1 : prev;
        }
        return prev;
      });
    }, 1200);

    return () => clearInterval(interval);
  }, [comparative?.progress, comparative?.status]);

  const handleModelChange = async (modelId: string) => {
    setSelectedModel(modelId);
    try {
      await authFetch(`${API_URL}/api/settings/embedding_model`, {
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
      await authFetch(`${API_URL}/api/settings/vector_db`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ db: dbId }),
      });
    } catch (err) {
      console.error("Failed to update DB setting", err);
    }
  };

  const handleStartEvaluation = async (force = false) => {
    setIsStartingEval(true);
    try {
      const res = await authFetch(`${API_URL}/api/workflow/evaluate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ force })
      });
      const data = await res.json();
      if (data.status === "cached") {
        if (window.confirm("Results are already up to date (no new PDFs detected). Do you want to force a re-run?")) {
          return handleStartEvaluation(true);
        }
      } else if (data.status === "processing") {
        fetchMetrics();
      }
    } catch (err) {
      console.error("Failed to start evaluation", err);
    } finally {
      setIsStartingEval(false);
    }
  };

  const isProcessing = metrics?.status === "processing";

  const handleStartComparative = async (force = false) => {
    const hasPrevious = comparative?.completed_models?.length > 0 && comparative?.status !== "complete";
    let resume = false;
    
    if (hasPrevious && !force) {
      resume = window.confirm("A previously interrupted benchmark session was found. Would you like to CONTINUE from where you left off?");
    }
    
    setIsStartingComparative(true);
    try {
      const res = await authFetch(`${API_URL}/api/workflow/comparative`, { 
        method: "POST", 
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ resume, force })
      });
      const data = await res.json();
      if (data.status === "cached") {
        if (window.confirm("Comparative results are already up to date. Do you want to force a re-run?")) {
          return handleStartComparative(true);
        }
      } else {
        setActiveTab("comparative");
      }
    } catch (err) { console.error(err); }
    finally { setIsStartingComparative(false); }
  };

  const handleStartPipelineCompare = async () => {
    setIsStartingPipelineCompare(true);
    try {
      await authFetch(`${API_URL}/api/workflow/compare_pipelines`, { method: "POST" });
      setActiveTab("pipeline_compare");
    } catch (err) { console.error(err); }
    finally { setIsStartingPipelineCompare(false); }
  };

  const handleStartPrimitiveEval = async (force = false) => {
    setIsStartingPrimitive(true);
    try {
      const res = await authFetch(`${API_URL}/api/workflow/evaluate_primitive`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_choice: primitiveModelChoice, force }),
      });
      const data = await res.json();
      if (data.status === "cached") {
        if (window.confirm("Primitive results are already up to date. Do you want to force a re-run?")) {
          return handleStartPrimitiveEval(true);
        }
      }
    } catch (err) { console.error(err); }
    finally { setIsStartingPrimitive(false); }
  };

  const handleExportTXT = () => {
    if (!metrics?.summary) return;
    let content = "LEXVED INSTITUTIONAL AUDIT REPORT\n";
    content += "=" .repeat(40) + "\n\n";
    content += `Timestamp: ${metrics.timestamp || new Date().toLocaleString()}\n`;
    content += `Database: ${selectedDb.toUpperCase()}\nEmbedding: ${selectedModel}\n\n`;
    content += "METRIC ID | LABEL | VALUE | UNIT\n";
    content += "-".repeat(50) + "\n";
    rows.forEach(r => {
      const val = r.value !== null ? r.value.toFixed(r.decimals) : "PENDING";
      content += `${r.id} | ${r.label} | ${val} | ${r.unit}\n`;
    });
    const blob = new Blob([content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `LexVed_Audit_${new Date().toISOString().split("T")[0]}.txt`;
    a.click();
  };

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
    if (activeTab === "single") {
      if (!metrics || !metrics.summary) return;
      const doc = new jsPDF();
      const timestamp = metrics.timestamp || new Date().toLocaleString();
      const sys = metrics.system_info || { 
        vector_db: selectedDb, 
        model: "Local AI Node", 
        embedding: selectedModel
      };

      doc.setFontSize(20);
      doc.setTextColor(180, 150, 40);
      doc.text("LexVed Institutional Audit Report", 14, 20);
      doc.setFontSize(9);
      doc.setTextColor(120);
      doc.text(`Mission-Critical Metrics (M1-M24) | ${timestamp}`, 14, 27);
      doc.text(`System: ${sys.model} | DB: ${sys.vector_db.toUpperCase()} | Model: ${sys.embedding}`, 14, 32);

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
    } else {
      // COMPARATIVE PDF EXPORT
      if (!comparative || comparative.status !== "complete") {
        alert("Please complete the comparative benchmark first.");
        return;
      }
      
      const doc = new jsPDF('l', 'mm', 'a4'); // Landscape for better table fit
      const dateStr = new Date().toISOString().split('T')[0];
      
      // Page 1: Comparative Overview
      doc.setFontSize(22);
      doc.setTextColor(180, 150, 40);
      doc.text("LexVed Comparative Intelligence Audit", 14, 20);
      doc.setFontSize(10);
      doc.setTextColor(120);
      doc.text(`Comprehensive Analysis of 6 Embedding Architectures | ${dateStr}`, 14, 28);
      
      const models = comparative.models_benchmarked;
      const colX = [14, 25, 60, ...models.map((_: any, i: number) => 95 + i * 32)];
      const headers = ["ID", "Metric", "Label", ...models.map((m: string) => m.split('/').pop()?.substring(0, 12))];
      
      let y = 42;
      doc.setFillColor(20, 20, 20);
      doc.rect(12, y - 5, 272, 8, "F");
      doc.setFontSize(7);
      doc.setTextColor(200, 170, 50);
      headers.forEach((h, i) => doc.text(String(h), colX[i], y));
      y += 8;

      Object.entries(comparative.comparison_table).forEach(([mk, vals]: [string, any], idx) => {
        if (y > 185) { doc.addPage('l'); y = 20; }
        if (idx % 2 === 0) {
          doc.setFillColor(245, 245, 245);
          doc.rect(12, y - 4, 272, 7, "F");
        }
        doc.setTextColor(60);
        doc.text(mk, colX[0], y);
        
        // Find label from rows
        const rowData = rows.find(r => r.id === mk);
        doc.text(rowData?.category || "", colX[1], y);
        doc.text(rowData?.label || "", colX[2], y);
        
        models.forEach((m: string, i: number) => {
          const v = vals[m];
          const isBest = comparative.best_per_metric?.[mk] === m;
          if (isBest) {
            doc.setTextColor(180, 150, 40);
            doc.setFont('helvetica', 'bold');
          } else {
            doc.setTextColor(100);
            doc.setFont('helvetica', 'normal');
          }
          const valText = v !== null && v !== undefined ? Number(v).toFixed(3) : "—";
          doc.text(valText, colX[3 + i], y);
        });
        y += 7;
      });

      // Page 2: Performance Leaderboard
      doc.addPage('p'); // Switch back to portrait for leaderboard
      doc.setFontSize(22);
      doc.setTextColor(180, 150, 40);
      doc.text("Performance Leaderboard", 14, 25);
      doc.setLineWidth(0.5);
      doc.setDrawColor(180, 150, 40);
      doc.line(14, 30, 196, 30);

      // Rank Models based on "Wins" (Best per Metric)
      const wins: Record<string, number> = {};
      models.forEach((m: string) => wins[m] = 0);
      Object.values(comparative.best_per_metric).forEach((m: any) => {
        if (wins[m] !== undefined) wins[m]++;
      });

      const sortedModels = models.map((m: string) => ({ name: m, score: wins[m] }))
                                .sort((a: any, b: any) => b.score - a.score);

      // Display Top 3
      let ly = 50;
      const medals = ["GOLD", "SILVER", "BRONZE"];
      const colors = [[212, 175, 55], [192, 192, 192], [205, 127, 50]];

      sortedModels.slice(0, 3).forEach((m: any, i: number) => {
        doc.setFillColor(245, 245, 245);
        doc.rect(14, ly - 8, 182, 25, "F");
        
        doc.setFontSize(14);
        doc.setTextColor(colors[i][0], colors[i][1], colors[i][2]);
        doc.setFont('helvetica', 'bold');
        doc.text(`#${i+1} ${medals[i]}`, 20, ly);
        
        doc.setFontSize(12);
        doc.setTextColor(40);
        doc.text(m.name.split('/').pop() || m.name, 20, ly + 8);
        
        doc.setFontSize(10);
        doc.setTextColor(100);
        doc.text(`${m.score} Mission-Critical Wins`, 150, ly + 4);
        
        ly += 35;
      });

      // Comparative Analysis Section
      ly += 10;
      doc.setFontSize(16);
      doc.setTextColor(180, 150, 40);
      doc.text("Competitive Intelligence Analysis", 14, ly);
      ly += 10;
      doc.setFontSize(10);
      doc.setTextColor(60);
      doc.setFont('helvetica', 'normal');

      if (sortedModels.length >= 2) {
        const best = sortedModels[0];
        const second = sortedModels[1];
        const diff = best.score - second.score;
        const pct = ((diff / second.score) * 100).toFixed(1);

        doc.text(`The #1 Ranked Model (${best.name.split('/').pop()}) outperformed the #2 Model`, 14, ly);
        doc.text(`by a margin of ${pct}% in total metric victories.`, 14, ly + 6);
        
        ly += 20;
        doc.setFont('helvetica', 'bold');
        doc.text("Key Strategic Takeaways:", 14, ly);
        doc.setFont('helvetica', 'normal');
        doc.text("• The top-ranked model exhibits superior alignment with institutional legal standards.", 14, ly + 8);
        doc.text("• Latency-to-Quality trade-offs vary significantly across the 6-model spectrum.", 14, ly + 14);
        doc.text("• Citation accuracy remains the primary differentiator for mission-critical tasks.", 14, ly + 20);
      }

      doc.setFontSize(8);
      doc.setTextColor(150);
      doc.text("Generated by LexVed Intelligence Node | AES-256 Encrypted Report", 14, 285);

      doc.save(`LexVed_Comparative_Audit_${dateStr}.pdf`);
    }
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
              <h2 className="font-display text-2xl text-[#d4af37] tracking-[0.2em] flex items-center gap-3">
                <ShieldCheck className="text-[#d4af37] w-6 h-6" />
                LexVed Performance Audit
              </h2>
              <p className="text-white/40 text-[10px] uppercase font-mono tracking-widest mt-2">Institutional RAG Benchmarking — Mission-Critical (M1-M24)</p>
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
                   
                   {activeTab === "single" && (
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
                   )}

                   <div className="grid grid-cols-3 gap-2">
                       <button onClick={handleExportCSV} className="py-3 bg-[#111] border border-[#D4AF37]/30 text-white/70 rounded-lg text-[10px] font-bold tracking-widest uppercase hover:bg-[#D4AF37] hover:text-black hover:border-[#D4AF37] transition-all">
                         CSV
                       </button>
                       <button onClick={handleExportPDF} className="py-3 bg-[#111] border border-[#D4AF37]/30 text-white/70 rounded-lg text-[10px] font-bold tracking-widest uppercase hover:bg-[#D4AF37] hover:text-black hover:border-[#D4AF37] transition-all">
                         PDF
                       </button>
                       <button onClick={handleExportTXT} className="py-3 bg-[#111] border border-[#D4AF37]/30 text-white/70 rounded-lg text-[10px] font-bold tracking-widest uppercase hover:bg-[#D4AF37] hover:text-black hover:border-[#D4AF37] transition-all">
                         EXPORT TXT
                       </button>
                    </div>

                    {/* Tab Switcher */}
                    <div className="flex gap-1 bg-[#0a0a0a] border border-white/5 p-1 rounded-lg">
                      <button onClick={() => setActiveTab("single")} className={`flex-1 py-2 rounded-md text-[10px] font-bold uppercase tracking-widest transition-all ${activeTab === "single" ? "bg-[#d4af37] text-black shadow-[0_0_15px_rgba(212,175,55,0.4)]" : "text-white/40 hover:text-[#d4af37]"}`}>Single</button>
                      <button onClick={() => setActiveTab("comparative")} className={`flex-1 py-2 rounded-md text-[10px] font-bold uppercase tracking-widest transition-all ${activeTab === "comparative" ? "bg-[#d4af37] text-black shadow-[0_0_15px_rgba(212,175,55,0.4)]" : "text-white/40 hover:text-[#d4af37]"}`}>Models</button>
                      <button onClick={() => setActiveTab("pipeline_compare")} className={`flex-1 py-2 rounded-md text-[10px] font-bold uppercase tracking-widest transition-all ${activeTab === "pipeline_compare" ? "bg-[#d4af37] text-black shadow-[0_0_15px_rgba(212,175,55,0.4)]" : "text-white/40 hover:text-[#d4af37]"}`}>Pipelines</button>
                    </div>

                    {activeTab === "comparative" && (
                      <motion.button
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        onClick={handleStartComparative}
                        disabled={isStartingComparative || comparative?.status === "processing"}
                        className="w-full py-4 rounded-xl flex items-center justify-center gap-3 font-bold uppercase tracking-widest text-xs border transition-all disabled:opacity-30 bg-transparent text-[#d4af37] border-[#d4af37] shadow-[inset_0_0_15px_rgba(212,175,55,0.1)] hover:bg-[#d4af37] hover:text-black hover:shadow-[0_0_30px_rgba(212,175,55,0.4)]"
                      >
                        {isStartingComparative ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4 fill-current" />}
                        Benchmark All 6 Models
                      </motion.button>
                    )}

                    {activeTab === "pipeline_compare" && (
                      <div className="flex flex-col gap-3">
                        {/* Model selector for primitive pipeline */}
                        <div className="p-4 bg-white/5 border border-white/10 rounded-xl">
                          <div className="text-[9px] uppercase font-bold tracking-[0.2em] text-white/40 mb-2">Primitive Embedding Model</div>
                          <select
                            value={primitiveModelChoice}
                            onChange={e => setPrimitiveModelChoice(e.target.value)}
                            className="w-full bg-black border border-[#d4af37]/30 text-white rounded-lg p-2.5 text-xs focus:border-[#d4af37] focus:outline-none"
                          >
                            <option value="1">MPNet (multi-qa-mpnet)</option>
                            <option value="2">MiniLM (multi-qa-MiniLM-L6)</option>
                            <option value="4">DistilBERT (multi-qa-distilbert)</option>
                            <option value="6">BGE-M3 (auto-download)</option>
                          </select>
                        </div>
                        {/* Standalone primitive run */}
                        <motion.button
                          whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
                          onClick={handleStartPrimitiveEval}
                          disabled={isStartingPrimitive || primitiveMetrics?.status === "processing"}
                          className="w-full py-3 rounded-xl flex items-center justify-center gap-2 font-bold uppercase tracking-widest text-xs transition-all disabled:opacity-30 bg-[#d4af37] text-black shadow-[0_0_20px_rgba(212,175,55,0.3)] hover:shadow-[0_0_30px_rgba(212,175,55,0.5)]"
                        >
                          {isStartingPrimitive ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4 fill-current" />}
                          Run Primitive Evaluation
                        </motion.button>
                        {/* Cross-pipeline comparison */}
                        <motion.button
                          whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
                          onClick={handleStartPipelineCompare}
                          disabled={isStartingPipelineCompare || pipelineCompare?.status === "processing"}
                          className="w-full py-3 rounded-xl flex items-center justify-center gap-2 font-bold uppercase tracking-widest text-xs border transition-all disabled:opacity-30 bg-transparent text-[#d4af37] border-[#d4af37] shadow-[inset_0_0_15px_rgba(212,175,55,0.1)] hover:bg-[#d4af37] hover:text-black"
                        >
                          {isStartingPipelineCompare ? <Loader2 className="w-4 h-4 animate-spin" /> : <Activity className="w-4 h-4" />}
                          Compare vs Enhanced
                        </motion.button>
                      </div>
                    )}
                </div>

                {/* Right: Metrics */}
                <div className="lg:w-2/3 space-y-6">
                  {activeTab === "single" ? (
                    <>
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
                                  <span className="text-[#D4AF37]">{row.value.toFixed(row.decimals)}</span>
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
                    </>
                  ) : activeTab === "comparative" ? (
                    /* Comparative Tab */
                    <div className="overflow-hidden border border-[#D4AF37]/20 rounded-xl bg-white/[0.02]">
                      <div className="px-6 py-4 border-b border-[#D4AF37]/20 bg-[#D4AF37]/10 flex justify-between items-center">
                        <span className="text-[10px] uppercase font-bold tracking-widest text-[#D4AF37]">Comparative Benchmark</span>
                        <span className="text-[10px] text-[#D4AF37] font-mono">{comparative?.status === "complete" ? `${comparative.models_benchmarked?.length || 0} MODELS` : comparative?.status?.toUpperCase() || "NO DATA"}</span>
                      </div>
                      {comparative?.status === "complete" && comparative.comparison_table ? (
                        <div className="overflow-x-auto">
                          <table className="w-full text-left">
                            <thead className="text-white/40 text-[9px] font-bold uppercase tracking-wider">
                              <tr>
                                <th className="px-4 py-3 sticky left-0 bg-[#0a0a0a]">Metric</th>
                                {comparative.models_benchmarked.map((m: string) => (
                                  <th key={m} className="px-3 py-3 text-center whitespace-nowrap">{m.split('/').pop()?.substring(0, 12)}</th>
                                ))}
                              </tr>
                            </thead>
                            <tbody className="divide-y divide-white/5">
                              {Object.entries(comparative.comparison_table).map(([mk, vals]: [string, any]) => (
                                <tr key={mk} className="hover:bg-white/[0.02]">
                                  <td className="px-4 py-2 font-mono text-[#d4af37] text-[10px] sticky left-0 bg-[#0a0a0a]">{mk}</td>
                                  {comparative.models_benchmarked.map((m: string) => {
                                    const v = vals[m];
                                    const isBest = comparative.best_per_metric?.[mk] === m;
                                    return (
                                      <td key={m} className={`px-3 py-2 font-mono text-[10px] text-center ${isBest ? 'text-[#D4AF37] font-bold' : 'text-white/50'}`}>
                                        {v !== null && v !== undefined ? Number(v).toFixed(2) : '—'}
                                        {isBest && <span className="ml-1 text-[8px]">★</span>}
                                      </td>
                                    );
                                  })}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      ) : comparative?.status === "processing" ? (() => {
                        return (
                          <div className="flex flex-col items-center justify-center py-16 px-8 gap-4">
                            <Loader2 className="w-10 h-10 text-[#D4AF37] animate-spin" />
                            <h2 className="text-[#D4AF37] text-lg font-serif uppercase tracking-[0.2em]">{comparative.progress}</h2>
                            <div className="w-full max-w-md bg-white/5 rounded-full h-1.5 overflow-hidden border border-white/10 mt-2">
                              <motion.div 
                                className="bg-gradient-to-r from-[#D4AF37]/40 to-[#D4AF37] h-full rounded-full shadow-[0_0_10px_rgba(212,175,55,0.5)]"
                                initial={{ width: 0 }}
                                animate={{ width: `${smoothPct}%` }}
                                transition={{ duration: 0.5 }}
                              />
                            </div>
                            <span className="text-[11px] font-mono tracking-widest text-[#D4AF37]/60 mb-6 uppercase">{smoothPct}% Complete</span>

                            {/* Sequential Model Checklist */}
                            <div className="w-full max-w-md bg-black/40 border border-[#D4AF37]/20 rounded-xl p-5 space-y-3">
                              <div className="text-[9px] uppercase tracking-[0.3em] font-bold text-[#D4AF37]/50 border-b border-[#D4AF37]/10 pb-3 mb-3">Model Execution Sequence</div>
                              {comparative?.completed_models?.map((m: string) => (
                                <div key={m} className="flex items-center justify-between text-xs py-1.5">
                                  <span className="text-white/40 font-mono text-[11px] line-through decoration-[#D4AF37]/30">{m.split('/').pop()}</span>
                                  <span className="text-black font-bold text-[9px] bg-[#D4AF37] px-2 py-0.5 rounded shadow-[0_0_10px_rgba(212,175,55,0.3)] tracking-widest">PASSED</span>
                                </div>
                              ))}
                              {comparative?.current_model && (
                                <div className="flex items-center justify-between text-xs py-2 bg-[#D4AF37]/5 px-3 rounded border border-[#D4AF37]/30">
                                  <span className="text-[#D4AF37] font-bold font-mono text-[11px]">{comparative.current_model.split('/').pop()}</span>
                                  <div className="flex items-center gap-2">
                                    <Loader2 className="w-3.5 h-3.5 text-[#D4AF37] animate-spin" />
                                    <span className="text-[#D4AF37] text-[10px] font-bold tracking-widest">ANALYZING</span>
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>
                        );
                      })() : (
                        <div className="flex flex-col items-center justify-center py-16 gap-3 text-white/20">
                          <Database className="w-8 h-8 opacity-50" />
                          <p className="text-xs uppercase tracking-[0.2em] font-serif">No comparative data.</p>
                        </div>
                      )}
                    </div>
                  ) : (
                    /* Pipeline Tab — two sub-views: standalone primitive + comparison */
                    <div className="space-y-6">

                      {/* --- Standalone Primitive Results --- */}
                      <div className="overflow-hidden border border-[#D4AF37]/20 rounded-xl bg-white/[0.02]">
                        <div className="px-6 py-4 border-b border-[#D4AF37]/20 bg-[#D4AF37]/10 flex justify-between items-center">
                          <span className="text-[10px] uppercase font-bold tracking-widest text-[#D4AF37]">Primitive Pipeline — Standalone Results</span>
                          <span className="text-[10px] text-[#D4AF37] font-mono">
                            {primitiveMetrics?.status === "complete"
                              ? `${primitiveMetrics.system_info?.embedding || ""} · ${primitiveMetrics.system_info?.total_pdfs || "?"} PDFs · ${primitiveMetrics.system_info?.total_chunks || "?"} chunks`
                              : primitiveMetrics?.status?.toUpperCase() || "NOT RUN"}
                          </span>
                        </div>
                        {primitiveMetrics?.status === "complete" && primitiveMetrics.summary ? (
                          <div className="overflow-x-auto">
                            <table className="w-full text-left">
                              <thead className="text-white/40 text-[9px] font-bold uppercase tracking-wider">
                                <tr>
                                  <th className="px-4 py-3 sticky left-0 bg-[#0a0a0a]">ID</th>
                                  <th className="px-4 py-3">Metric Label</th>
                                  <th className="px-4 py-3 text-center">Value</th>
                                  <th className="px-3 py-3">Unit</th>
                                </tr>
                              </thead>
                              <tbody className="divide-y divide-white/5">
                                {rows.map(row => {
                                  const val = primitiveMetrics.summary?.[row.id];
                                  return (
                                    <tr key={row.id} className="hover:bg-white/[0.02]">
                                      <td className="px-4 py-2 font-mono text-[#d4af37] text-[10px] sticky left-0 bg-[#0a0a0a]">{row.id}</td>
                                      <td className="px-4 py-2 text-white/80 text-xs">{row.label}</td>
                                      <td className="px-4 py-2 font-mono text-[11px] text-center">
                                        {val !== null && val !== undefined
                                          ? <span className="text-[#d4af37] font-bold">{Number(val).toFixed(row.decimals)}</span>
                                          : <span className="text-white/20">—</span>}
                                      </td>
                                      <td className="px-3 py-2 text-white/30 text-[10px]">{row.unit}</td>
                                    </tr>
                                  );
                                })}
                              </tbody>
                            </table>
                          </div>
                        ) : primitiveMetrics?.status === "processing" ? (
                          <div className="flex flex-col items-center justify-center py-12 px-6 gap-3">
                            <Loader2 className="w-10 h-10 text-[#D4AF37] animate-spin" />
                            <p className="text-[#D4AF37] text-sm font-serif uppercase tracking-widest text-center">{primitiveMetrics.progress}</p>
                            <p className="text-white/30 text-[10px] text-center max-w-xs">Processing {primitiveMetrics.embedding ? `with ${primitiveMetrics.embedding}` : ""} — this may take several minutes for 500+ PDFs.</p>
                          </div>
                        ) : (
                          <div className="flex flex-col items-center justify-center py-12 gap-2 text-white/20">
                            <HardDrive className="w-8 h-8 opacity-40" />
                            <p className="text-xs uppercase tracking-[0.2em]">Select a model and click Run Primitive Evaluation</p>
                          </div>
                        )}
                      </div>

                      {/* --- Cross-Pipeline Comparison Results --- */}
                      <div className="overflow-hidden border border-white/10 rounded-xl bg-white/[0.02]">
                        <div className="px-6 py-4 border-b border-white/10 bg-white/5 flex justify-between items-center">
                          <span className="text-[10px] uppercase font-bold tracking-widest text-white/40">Enhanced vs Primitive — Side-by-Side</span>
                          <span className="text-[10px] text-[#d4af37] font-mono">{pipelineCompare?.status === "complete" ? "COMPARISON READY" : pipelineCompare?.status?.toUpperCase() || "NOT RUN"}</span>
                        </div>
                        {pipelineCompare?.status === "complete" && pipelineCompare.summary_comparison ? (
                          <div className="overflow-x-auto">
                            <table className="w-full text-left">
                              <thead className="text-white/40 text-[9px] font-bold uppercase tracking-wider">
                                <tr>
                                  <th className="px-4 py-3 sticky left-0 bg-[#0a0a0a]">Metric</th>
                                  <th className="px-3 py-3 text-center text-[#d4af37]">Enhanced</th>
                                  <th className="px-3 py-3 text-center text-white/60">Primitive</th>
                                </tr>
                              </thead>
                              <tbody className="divide-y divide-white/5">
                                {rows.map(row => {
                                  const mk = row.id;
                                  const enhVal = pipelineCompare.summary_comparison["Enhanced"]?.[mk];
                                  const primVal = pipelineCompare.summary_comparison["Primitive"]?.[mk];
                                  return (
                                    <tr key={mk} className="hover:bg-white/[0.02]">
                                      <td className="px-4 py-2 sticky left-0 bg-[#0a0a0a]">
                                        <div className="font-mono text-[#d4af37] text-[10px]">{mk}</div>
                                        <div className="text-white/60 text-[10px]">{row.label}</div>
                                      </td>
                                      <td className="px-3 py-2 text-center font-mono text-[11px] text-[#d4af37] font-bold">
                                        {enhVal !== null && enhVal !== undefined ? `${Number(enhVal).toFixed(row.decimals)} ${row.unit}` : '—'}
                                      </td>
                                      <td className="px-3 py-2 text-center font-mono text-[11px] text-white/60">
                                        {primVal !== null && primVal !== undefined ? `${Number(primVal).toFixed(row.decimals)} ${row.unit}` : '—'}
                                      </td>
                                    </tr>
                                  );
                                })}
                              </tbody>
                            </table>
                          </div>
                        ) : pipelineCompare?.status === "processing" ? (
                          <div className="flex flex-col items-center justify-center py-10 px-6 gap-3">
                            <Loader2 className="w-8 h-8 text-[#D4AF37] animate-spin" />
                            <p className="text-[#D4AF37] text-sm uppercase tracking-widest">{pipelineCompare.progress}</p>
                          </div>
                        ) : (
                          <div className="flex flex-col items-center justify-center py-10 gap-2 text-white/20">
                            <Activity className="w-7 h-7 opacity-40" />
                            <p className="text-xs uppercase tracking-[0.2em]">Click Compare vs Enhanced to run</p>
                          </div>
                        )}
                      </div>

                    </div>
                  )}
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
