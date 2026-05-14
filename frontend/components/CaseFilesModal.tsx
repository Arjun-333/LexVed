"use client";

import React, { useState, useEffect } from "react";
import { X, Folder, FileText, Database, Shield } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useAuth } from "./AuthContext";

interface FileItem {
  name: string;
  size: string;
  type: string;
}

export default function CaseFilesModal({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const { authFetch } = useAuth();
  const [files, setFiles] = useState<FileItem[]>([]);
  const [loading, setLoading] = useState(true);
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:5000";

  useEffect(() => {
    if (isOpen) {
      setLoading(true);
      authFetch(`${API_URL}/api/files`)
        .then((res) => res.json())
        .then((data) => { setFiles(data); setLoading(false); })
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
                <Folder className="text-[#D4AF37] w-5 h-5" />
                Case Files & Legal Corpus
              </h2>
              <p className="text-[10px] uppercase font-mono tracking-widest mt-2" style={{ color: 'var(--text-muted)' }}>Authorized LexVed document repository</p>
            </div>
            <div className="flex items-center gap-3">
              <label className="cursor-pointer bg-[#D4AF37]/10 border border-[#D4AF37]/30 text-[#D4AF37] hover:bg-[#D4AF37] hover:text-black transition-all px-4 py-2 rounded-lg text-xs font-bold uppercase tracking-widest flex items-center gap-2">
                <FileText size={14} />
                Upload PDF
                <input 
                  type="file" 
                  className="hidden" 
                  accept=".pdf" 
                  onChange={async (e) => {
                    const file = e.target.files?.[0];
                    if (!file) return;
                    setLoading(true);
                    const formData = new FormData();
                    formData.append("file", file);
                    try {
                      await authFetch(`${API_URL}/api/ingest`, { method: "POST", body: formData });
                      const res = await authFetch(`${API_URL}/api/files`);
                      setFiles(await res.json());
                    } catch (err) { console.error(err); }
                    setLoading(false);
                  }} 
                />
              </label>
              <button onClick={onClose} className="p-2 rounded-full transition-colors" style={{ color: 'var(--text-muted)' }}>
                <X />
              </button>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
            {loading ? (
              <div className="flex flex-col items-center justify-center py-20 gap-4">
                <div className="w-8 h-8 border-2 border-[#D4AF37]/30 border-t-[#D4AF37] rounded-full animate-spin" />
                <p className="text-sm animate-pulse uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>Scanning Repository...</p>
              </div>
            ) : (
              <div className="grid gap-2">
                {files.map((file, idx) => (
                  <motion.div 
                    key={idx}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.05 }}
                    className="group flex items-center gap-4 p-4 rounded-xl transition-all cursor-default"
                    style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}
                  >
                    <div className="w-10 h-10 rounded-lg bg-[#D4AF37]/10 flex items-center justify-center text-[#D4AF37] group-hover:scale-110 transition-transform">
                      {file.type === "Knowledge Base" ? <Database size={18} /> : (file.type === "Evaluation Suite" ? <Shield size={18} /> : <FileText size={18} />)}
                    </div>
                    <div className="flex-1 min-w-0">
                      <h4 className="text-sm font-medium truncate" style={{ color: 'var(--text)' }}>{file.name}</h4>
                      <p className="text-[10px] uppercase tracking-wider mt-0.5" style={{ color: 'var(--text-muted)' }}>{file.type} • {file.size}</p>
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
          </div>

          <div className="p-4 flex items-center justify-between" style={{ background: 'var(--surface)', borderTop: '1px solid var(--border)' }}>
            <div className="flex items-center gap-2 text-[10px] uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>
              <Shield className="w-3 h-3 text-[#D4AF37]/50" />
              AES-256 Vaulted Storage
            </div>
            <div className="text-[10px] italic" style={{ color: 'var(--text-muted)' }}>
              {files.length} legal assets indexed
            </div>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
