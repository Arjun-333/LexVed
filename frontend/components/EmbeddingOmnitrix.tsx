"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

const embeddingModels = [
  { id: "multi-qa-mpnet-base-cos-v1", name: "MPNet Base", icon: "hub", color: "var(--accent)" },
  { id: "multi-qa-MiniLM-L6-cos-v1", name: "MiniLM L6", icon: "architecture", color: "#60A5FA" },
  { id: "multi-qa-distilbert-cos-v1", name: "DistilBERT", icon: "psychology", color: "#F472B6" },
  { id: "BAAI/bge-m3", name: "BGE-M3", icon: "language", color: "#34D399" },
  { id: "intfloat/multilingual-e5-large-instruct", name: "E5-Mistral", icon: "auto_awesome", color: "#FB923C" },
  { id: "Cohere/Cohere-embed-english-v3.0", name: "Cohere v3", icon: "cloud", color: "#A78BFA" },
];

interface Props {
  selectedModel: string;
  onSelect: (id: string) => void;
}

export default function EmbeddingOmnitrix({ selectedModel, onSelect }: Props) {
  const [rotation, setRotation] = useState(0);
  const DEG = 360 / embeddingModels.length;
  const currentIndex = embeddingModels.findIndex(m => m.id === selectedModel);
  
  // Snap to the selected model if changed from outside
  useEffect(() => {
    setRotation(-currentIndex * DEG);
  }, [currentIndex]);

  const handleRotate = (direction: number) => {
    const nextRotation = rotation + (direction * DEG);
    setRotation(nextRotation);
    
    // Calculate which model is now active
    // normalize rotation to [0, 360) and then find index
    const normalized = ((-nextRotation % 360) + 360) % 360;
    const index = Math.round(normalized / DEG) % embeddingModels.length;
    onSelect(embeddingModels[index].id);
  };

  return (
    <div className="flex flex-col items-center gap-6 p-8 bg-[var(--surface)] border border-[var(--border)] rounded-3xl shadow-2xl relative overflow-hidden group">
      {/* Glow Effect */}
      <div className="absolute inset-0 bg-[var(--accent-glow)] opacity-5 group-hover:opacity-10 transition-opacity pointer-events-none" />
      
      <h3 className="text-[10px] font-bold uppercase tracking-[0.3em] text-[var(--accent)] opacity-80">
        Embedding Omnitrix
      </h3>

      {/* The Dial */}
      <div className="relative w-48 h-48 flex items-center justify-center">
        {/* Outer Ring */}
        <div className="absolute inset-0 rounded-full border-4 border-[var(--border)] border-t-[var(--accent)] animate-[spin_10s_linear_infinite] opacity-20" />
        
        {/* Core Dial */}
        <motion.div 
          className="relative w-36 h-36 rounded-full bg-[var(--bg-secondary)] border-2 border-[var(--accent-glow)] shadow-[0_0_20px_rgba(212,175,55,0.2)] flex items-center justify-center cursor-grab active:cursor-grabbing"
          animate={{ rotate: rotation }}
          transition={{ type: "spring", stiffness: 100, damping: 15 }}
        >
          {/* Inner Marks */}
          {embeddingModels.map((_, i) => (
            <div 
              key={i}
              className="absolute w-1 h-3 bg-[var(--accent)] rounded-full"
              style={{ 
                transform: `rotate(${i * DEG}deg) translateY(-60px)`,
                opacity: currentIndex === i ? 1 : 0.2
              }}
            />
          ))}

          {/* Icons on the Dial */}
          {embeddingModels.map((m, i) => (
            <div 
              key={m.id}
              className="absolute flex items-center justify-center"
              style={{ 
                transform: `rotate(${i * DEG}deg) translateY(-40px) rotate(${-i * DEG - rotation}deg)`,
                opacity: currentIndex === i ? 1 : 0.1,
                scale: currentIndex === i ? 1.2 : 0.8,
                transition: "all 0.3s ease"
              }}
            >
              <span className="material-icons-round text-2xl" style={{ color: m.color }}>{m.icon}</span>
            </div>
          ))}

          {/* Center Jewel */}
          <div className="w-12 h-12 rounded-full bg-gradient-to-br from-[var(--accent)] to-[var(--accent-dim)] shadow-[inset_0_0_10px_rgba(0,0,0,0.5),0_0_15px_var(--accent-glow)] flex items-center justify-center overflow-hidden">
             <div className="w-full h-full bg-[url('https://www.transparenttextures.com/patterns/carbon-fibre.png')] opacity-20" />
             <div className="absolute w-6 h-1 bg-white/30 rounded-full blur-[1px] -rotate-45 translate-x-1 -translate-y-1" />
          </div>
        </motion.div>

        {/* Interaction Overlays */}
        <div className="absolute inset-0 flex items-center justify-between px-2 pointer-events-none">
           <button 
             onClick={() => handleRotate(1)} 
             className="w-8 h-8 rounded-full bg-black/50 border border-white/10 flex items-center justify-center text-white pointer-events-auto hover:bg-[var(--accent)] transition-colors"
           >
             <span className="material-icons-round">chevron_left</span>
           </button>
           <button 
             onClick={() => handleRotate(-1)} 
             className="w-8 h-8 rounded-full bg-black/50 border border-white/10 flex items-center justify-center text-white pointer-events-auto hover:bg-[var(--accent)] transition-colors"
           >
             <span className="material-icons-round">chevron_right</span>
           </button>
        </div>
      </div>

      <div className="text-center z-10">
        <AnimatePresence mode="wait">
          <motion.div
            key={selectedModel}
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -5 }}
            className="flex flex-col items-center"
          >
            <span className="text-white font-bold text-sm tracking-wide">
              {embeddingModels[currentIndex].name}
            </span>
            <span className="text-[9px] uppercase tracking-widest text-[var(--accent)] font-bold mt-1">
              {selectedModel.includes("MiniLM") ? "384 DIM" : selectedModel.includes("bge-m3") || selectedModel.includes("e5-large") || selectedModel.includes("Cohere") ? "1024 DIM" : "768 DIM"}
            </span>
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
