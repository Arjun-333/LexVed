"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

const modelsList = [
  { id: "llama3", name: "LLaMA-3 8B (Instruct)",   icon: "security",      color: "#D4A017" },
  { id: "llama3:70b", name: "LLaMA-3 70B (Instruct)",   icon: "all_inclusive", color: "#8B5CF6" },
  { id: "qwen2.5:70b", name: "Qwen2.5-70B (Instruct)",   icon: "auto_awesome", color: "#EF4444" },
  { id: "qwen2.5:7b", name: "Qwen2.5-7B (Instruct)",    icon: "star",         color: "#F59E0B" },
  { id: "mistral", name: "Mistral 7B (Instruct)",    icon: "bolt",         color: "#10B981" },
  { id: "phi3", name: "Phi-3",                    icon: "memory",       color: "#06B6D4" },
];

const DEG = 360 / modelsList.length;
const R = 150;

export default function ModelWheel() {
  const [rotation, setRotation] = useState(0);
  const [isFocused, setIsFocused] = useState(false);
  const [activeModelId, setActiveModelId] = useState("llama3");
  const dragging = useRef(false);
  const lastY = useRef(0);
  const vel = useRef(0);

  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:5000";

  useEffect(() => {
    fetch(`${API_URL}/api/settings/generation_model`)
      .then(res => res.json())
      .then(data => setActiveModelId(data.model))
      .catch(console.error);
  }, []);

  const handleModelClick = async (id: string, idx: number) => {
    setActiveModelId(id);
    setRotation(idx * DEG);
    try {
      await fetch(`${API_URL}/api/settings/generation_model`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: id }),
      });
    } catch (err) {
      console.error(err);
    }
  };

  const snap = (r: number) => Math.round(r / DEG) * DEG;

  return (
    <aside
      onMouseEnter={() => setIsFocused(true)}
      onMouseLeave={() => setIsFocused(false)}
      className="w-[280px] hidden xl:flex flex-col items-center justify-center py-8 px-5 z-40 select-none transition-all duration-700"
      style={{
        background: "var(--bg-secondary)",
        borderLeft: "1px solid var(--border)",
        opacity: isFocused ? 1 : 0.6,
      }}
    >
      <div className="text-center mb-10 transition-all duration-500" style={{ transform: isFocused ? "scale(1)" : "scale(0.95)" }}>
        <h3
          className="text-[0.6rem] font-bold uppercase tracking-[0.25em] mb-1.5"
          style={{ color: "var(--accent)" }}
        >
          Intelligence Engine
        </h3>
        <p className="text-[0.55rem] font-bold uppercase tracking-[0.1em] opacity-40 transition-colors duration-300" style={{ color: "var(--text)" }}>
          LOCAL CLUSTER STATUS: <span className="text-green-500">OPTIMAL</span>
        </p>
      </div>

      {/* 3D Viewport */}
      <div
        className="w-[240px] h-[340px] relative overflow-hidden cursor-grab active:cursor-grabbing group"
        style={{ perspective: "1000px" }}
        onMouseDown={(e) => { dragging.current = true; lastY.current = e.clientY; vel.current = 0; }}
        onMouseMove={(e) => {
          if (!dragging.current) return;
          const d = e.clientY - lastY.current; lastY.current = e.clientY;
          const rd = -(d * 0.4); vel.current = rd;
          setRotation((r) => r + rd);
        }}
        onMouseUp={() => { dragging.current = false; setRotation((r) => snap(r + vel.current * 4)); }}
        onMouseLeave={() => { if (dragging.current) { dragging.current = false; setRotation((r) => snap(r + vel.current * 4)); }}}
        onWheel={(e) => { e.preventDefault(); setRotation((r) => snap(r + (e.deltaY > 0 ? DEG : -DEG))); }}
      >
        <div className="absolute top-0 left-0 right-0 h-32 z-10 pointer-events-none"
          style={{ background: "linear-gradient(to bottom, var(--bg-secondary), transparent)" }}
        />

        <div
          className="w-full h-full absolute"
          style={{
            transformStyle: "preserve-3d",
            transform: `rotateX(${rotation}deg)`,
            transition: dragging.current ? "none" : "transform 0.9s cubic-bezier(0.16, 1, 0.3, 1)",
          }}
        >
          {modelsList.map((m, i) => {
            const isActive = activeModelId === m.id;
            return (
              <div
                key={m.name}
                onClick={() => handleModelClick(m.id, i)}
                className="absolute w-[200px] h-[85px] left-[20px] top-[127px] rounded-2xl flex flex-col items-center justify-center gap-1.5 p-4 transition-all duration-500 cursor-pointer"
                style={{
                  transform: `rotateX(${-i * DEG}deg) translateZ(${R}px)`,
                  backfaceVisibility: "hidden",
                  background: "var(--surface)",
                  border: "1px solid var(--border)",
                  borderTopColor: isActive ? "var(--accent)" : "var(--border)",
                  boxShadow: isActive ? "var(--shadow-gold)" : "var(--shadow-prestige)",
                  opacity: isActive ? 1 : 0.4,
                }}
              >
                <span className="material-icons-round text-[22px]" style={{ color: m.color }}>
                  {m.icon}
                </span>
                <span className="font-bold text-[0.7rem] text-center tracking-tight" style={{ color: "var(--text)" }}>
                  {m.name}
                </span>
                <span className="text-[0.45rem] font-bold uppercase tracking-[0.2em] opacity-50">
                  {isActive ? "Active Link" : "Standby"}
                </span>
              </div>
            );
          })}
        </div>

        <div className="absolute bottom-0 left-0 right-0 h-32 z-10 pointer-events-none"
          style={{ background: "linear-gradient(to top, var(--bg-secondary), transparent)" }}
        />
      </div>

      <div className="flex gap-4 mt-10">
        <button
          onClick={() => setRotation((r) => snap(r - DEG))}
          className="w-10 h-10 rounded-full flex items-center justify-center cursor-pointer transition-all duration-300
            hover:bg-[var(--accent-bg)] border border-[var(--border)]"
          style={{ color: "var(--text-muted)" }}
        >
          <span className="material-icons-round text-[20px]">north</span>
        </button>
        <button
          onClick={() => setRotation((r) => snap(r + DEG))}
          className="w-10 h-10 rounded-full flex items-center justify-center cursor-pointer transition-all duration-300
            hover:bg-[var(--accent-bg)] border border-[var(--border)]"
          style={{ color: "var(--text-muted)" }}
        >
          <span className="material-icons-round text-[20px]">south</span>
        </button>
      </div>
    </aside>
  );
}
