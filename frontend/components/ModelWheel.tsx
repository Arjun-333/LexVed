"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

const modelsList = [
  { id: "ensemble", name: "Ensemble Logic", icon: "hub", color: "#D4AF37" },
  { id: "llama-3.3-70b-versatile", name: "Llama 3.3 70B", icon: "auto_awesome", color: "#D4AF37" },
  { id: "mixtral-8x7b-32768", name: "Mixtral 8x7B", icon: "all_inclusive", color: "#D4AF37" },
  { id: "mistral", name: "Mistral 7B", icon: "bolt", color: "#D4AF37" },
  { id: "llama-3.1-8b-instant", name: "Llama 3.1 8B", icon: "bolt", color: "#AA8C2C" },
  { id: "llama3", name: "Local Llama 3 8B", icon: "memory", color: "#AA8C2C" },
  { id: "qwen2.5:7b", name: "Local Qwen 2.5", icon: "security", color: "#AA8C2C" },
];

const DEG = 360 / modelsList.length;
const R = 150;

export default function ModelWheel({ 
  activeModelId, 
  onModelChange 
}: { 
  activeModelId: string; 
  onModelChange: (id: string) => void;
}) {
  const [rotation, setRotation] = useState(0);
  const [isFocused, setIsFocused] = useState(false);
  const [healthStatus, setHealthStatus] = useState("OPTIMAL");
  const dragging = useRef(false);
  const lastY = useRef(0);
  const vel = useRef(0);

  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:5000";

  useEffect(() => {
    // Sync initial rotation based on external activeModelId
    const idx = modelsList.findIndex(m => m.id === activeModelId);
    if (idx !== -1) setRotation(idx * DEG);
  }, [activeModelId]);

  useEffect(() => {
    // Dynamic Health Monitoring
    const checkHealth = () => {
      fetch(`${API_URL}/api/health`)
        .then(res => res.json())
        .then(data => {
          if (data.ollama === "connected" && data.vector_db === "connected") {
            setHealthStatus("OPTIMAL");
          } else if (data.ollama === "connected" || data.vector_db === "connected") {
            setHealthStatus("DEGRADED");
          } else {
            setHealthStatus("OFFLINE");
          }
        })
        .catch(() => setHealthStatus("OFFLINE"));
    };

    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  const snap = (r: number) => Math.round(r / DEG) * DEG;

  const handleModelClick = async (id: string, idx: number) => {
    onModelChange(id);
    
    // Continuous shortest-path rotation logic
    setRotation((prevRotation) => {
      const targetAngle = idx * DEG;
      // Find the multiple of 360 that brings the targetAngle closest to prevRotation
      const rounds = Math.round((prevRotation - targetAngle) / 360);
      return targetAngle + (rounds * 360);
    });
    
    try {
      const token = localStorage.getItem("lexved_token");
      const res = await fetch(`${API_URL}/api/settings/generation_model`, {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "Authorization": `Bearer ${token}`
        },
        body: JSON.stringify({ model: id }),
      });
      if (!res.ok) throw new Error("Failed to update model");
    } catch (err) {
      console.error(err);
    }
  };

  const stepModel = (direction: number) => {
    const currentIndex = modelsList.findIndex(m => m.id === activeModelId);
    let nextIndex = (currentIndex + direction) % modelsList.length;
    if (nextIndex < 0) nextIndex = modelsList.length - 1;
    
    const nextModel = modelsList[nextIndex];
    handleModelClick(nextModel.id, nextIndex);
  };

  return (
    <aside
      onMouseEnter={() => setIsFocused(true)}
      onMouseLeave={() => setIsFocused(false)}
      className="w-[280px] hidden xl:flex flex-col items-center justify-center py-8 px-5 z-40 select-none transition-all duration-700"
      style={{
        background: "var(--bg-secondary)",
        borderLeft: "1px solid var(--border)",
        opacity: 1.0, // Always visible
      }}
    >
      <div className="text-center mb-10 transition-all duration-500" style={{ transform: isFocused ? "scale(1)" : "scale(0.98)" }}>
        <h3
          className="text-[0.65rem] font-bold uppercase tracking-[0.25em] mb-1.5"
          style={{ color: "var(--accent)", textShadow: "0 0 10px rgba(212, 175, 55, 0.3)" }}
        >
          Intelligence Engine
        </h3>
        <p className="text-[0.6rem] font-bold uppercase tracking-[0.1em] opacity-60 transition-colors duration-300" style={{ color: "var(--text)" }}>
          LOCAL CLUSTER STATUS: <span style={{ 
            color: healthStatus === "OPTIMAL" ? "var(--accent)" : 
                   healthStatus === "DEGRADED" ? "#FFA500" : "#FF6B6B" ,
            textShadow: healthStatus === "OPTIMAL" ? "0 0 8px rgba(212, 175, 55, 0.4)" : "none"
          }}>{healthStatus}</span>
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
                key={m.id}
                onClick={() => handleModelClick(m.id, i)}
                className="absolute w-[200px] h-[85px] left-[20px] top-[127px] rounded-2xl flex flex-col items-center justify-center gap-1.5 p-4 transition-all duration-500 cursor-pointer"
                style={{
                  transform: `rotateX(${-i * DEG}deg) translateZ(${R}px)`,
                  backfaceVisibility: "hidden",
                  background: isActive ? "var(--surface-active)" : "var(--surface)",
                  borderStyle: "solid",
                  borderWidth: isActive ? "2px" : "1px",
                  borderColor: isActive ? "var(--accent)" : "var(--border)",
                  boxShadow: isActive ? "var(--shadow-gold), 0 0 20px rgba(212, 175, 55, 0.1)" : "var(--shadow-prestige)",
                  opacity: isActive ? 1 : 0.75, // Significantly increased from 0.4
                }}
              >
                <span className="material-icons-round text-[24px]" style={{ color: m.color, opacity: isActive ? 1 : 0.6 }}>
                  {m.icon}
                </span>
                <span className="font-bold text-[0.75rem] text-center tracking-tight" style={{ color: isActive ? "var(--accent)" : "var(--text)" }}>
                  {m.name}
                </span>
                <span className="text-[0.5rem] font-bold uppercase tracking-[0.2em] opacity-50">
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
          onClick={() => stepModel(-1)}
          className="w-10 h-10 rounded-full flex items-center justify-center cursor-pointer transition-all duration-300
            hover:bg-[var(--accent-bg)] border border-[var(--border)]"
          style={{ color: "var(--text-muted)" }}
        >
          <span className="material-icons-round text-[20px]">north</span>
        </button>
        <button
          onClick={() => stepModel(1)}
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
