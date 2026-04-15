"use client";

import React, { useState, useEffect } from "react";
import { X, Folder, FileText, Database, Shield } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface FileItem {
  name: string;
  size: string;
  type: string;
}

export default function CaseFilesModal({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const [files, setFiles] = useState<FileItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (isOpen) {
      setLoading(true);
      const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:5000";
      fetch(`${API_URL}/api/files`)
        .then((res) => res.json())
        .then((data) => {
          setFiles(data);
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
          <div className="p-6 border-b border-white/10 flex items-center justify-between bg-gradient-to-r from-blue-500/5 to-transparent">
            <div>
              <h2 className="text-xl font-bold text-white flex items-center gap-3">
                <Folder className="text-blue-400 w-5 h-5" />
                Case Files & Legal Corpus
              </h2>
              <p className="text-white/40 text-xs mt-1">Authorized LexVed document repository</p>
            </div>
            <button onClick={onClose} className="p-2 hover:bg-white/5 rounded-full transition-colors">
              <X className="text-white/40 hover:text-white" />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
            {loading ? (
              <div className="flex flex-col items-center justify-center py-20 gap-4">
                <div className="w-8 h-8 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin" />
                <p className="text-white/30 text-sm animate-pulse uppercase tracking-widest">Scanning Repository...</p>
              </div>
            ) : (
              <div className="grid gap-2">
                {files.map((file, idx) => (
                  <motion.div 
                    key={idx}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.05 }}
                    className="group flex items-center gap-4 p-4 rounded-xl border border-white/5 bg-white/[0.02] hover:bg-white/[0.05] hover:border-blue-500/30 transition-all cursor-default"
                  >
                    <div className="w-10 h-10 rounded-lg bg-blue-500/10 flex items-center justify-center text-blue-400 group-hover:scale-110 transition-transform">
                      {file.type === "Knowledge Base" ? <Database size={18} /> : (file.type === "Evaluation Suite" ? <Shield size={18} /> : <FileText size={18} />)}
                    </div>
                    <div className="flex-1 min-w-0">
                      <h4 className="text-sm font-medium text-white truncate">{file.name}</h4>
                      <p className="text-[10px] text-white/30 uppercase tracking-wider mt-0.5">{file.type} • {file.size}</p>
                    </div>
                    <div className="opacity-0 group-hover:opacity-100 transition-opacity">
                      <button 
                        onClick={() => {
                          const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:5000";
                          fetch(`${API_URL}/api/analyze`, {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ file: file.name })
                          }).then(() => alert("Mission-Critical Performance Audit Initiated. Check the Dashboard for live results."));
                        }}
                        className="text-[10px] font-bold text-blue-400 hover:text-white uppercase tracking-widest bg-blue-500/10 px-3 py-1.5 rounded-md border border-blue-500/20"
                      >
                        Analyze
                      </button>
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
          </div>

          <div className="p-4 bg-black/40 border-t border-white/5 flex items-center justify-between">
            <div className="flex items-center gap-2 text-[10px] text-white/30 uppercase tracking-widest">
              <Shield className="w-3 h-3 text-green-500/50" />
              AES-256 Vaulted Storage
            </div>
            <div className="text-[10px] text-white/20 italic">
              {files.length} legal assets indexed
            </div>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
