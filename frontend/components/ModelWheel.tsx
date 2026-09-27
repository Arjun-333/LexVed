"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useAuth } from "./AuthContext";

interface ModelMeta {
  id: string;
  name: string;
  shortName: string;
  icon: string;
  tier: string;
}

const MODEL_LIST: ModelMeta[] = [
  { id: "ensemble",               name: "Ensemble Logic",  shortName: "Ensemble", icon: "hub",           tier: "orchestrator" },
  { id: "llama3",                 name: "Llama 3 8B",      shortName: "L3-8B",    icon: "memory",        tier: "local" },
  { id: "llama-3.1-8b-instant",   name: "Llama 3.1 8B",   shortName: "L3.1",     icon: "bolt",          tier: "cloud" },
  { id: "llama3:70b",             name: "Llama 3 70B",     shortName: "L3-70B",   icon: "psychology",    tier: "local" },
  { id: "qwen2.5:70b",            name: "Qwen 2.5 70B",    shortName: "Q2.5-70B", icon: "auto_awesome",  tier: "local" },
  { id: "qwen2.5:7b",             name: "Qwen 2.5 7B",     shortName: "Q2.5-7B",  icon: "security",      tier: "local" },
  { id: "llama-3.3-70b-versatile",name: "Llama 3.3 70B",   shortName: "L3.3",     icon: "star",          tier: "cloud" },
  { id: "mixtral-8x7b-32768",     name: "Mixtral 8x7B",    shortName: "Mixtral",  icon: "all_inclusive", tier: "cloud" },
];

const N = MODEL_LIST.length;
const R = 110; // orbit radius in px
const R_OUTER = 195; // invisible outer rotation track radius in px
const TAB_W = 46; // width of the semicircle trigger tab
const TAB_H = 92; // height of the semicircle trigger tab
const ROTATE_SPEED = 95; // degrees per second while hovering

export default function ModelWheel({
  activeModelId,
  onModelChange,
}: {
  activeModelId: string;
  onModelChange: (id: string) => void;
}) {
  const { authFetch } = useAuth();
  const [isOpen, setIsOpen] = useState(false);
  const [hoveredModelId, setHoveredModelId] = useState<string | null>(null);
  const [rotationDeg, setRotationDeg] = useState(0);

  // Ref for detecting outside clicks
  const wrapperRef = useRef<HTMLElement>(null);

  // Refs for smooth rAF rotation with directional control
  const rotationRef = useRef(0);
  const rotateDirRef = useRef<1 | -1>(1); // 1 = down, -1 = up
  const isHoveringRef = useRef(false);
  const animFrameRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number | null>(null);

  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:5000";

  useEffect(() => {
    const checkHealth = () => authFetch(`${API_URL}/api/health`).catch(() => {});
    checkHealth();
    const interval = setInterval(checkHealth, 20000);
    return () => clearInterval(interval);
  }, [authFetch, API_URL]);

  // rAF rotation loop with directional support
  const tick = useCallback((timestamp: number) => {
    if (!isHoveringRef.current) return;
    if (lastTimeRef.current !== null) {
      const delta = timestamp - lastTimeRef.current;
      rotationRef.current = (rotationRef.current + rotateDirRef.current * (ROTATE_SPEED * delta) / 1000) % 360;
      setRotationDeg(rotationRef.current);
    }
    lastTimeRef.current = timestamp;
    animFrameRef.current = requestAnimationFrame(tick);
  }, []);

  const startRotation = useCallback(() => {
    isHoveringRef.current = true;
    lastTimeRef.current = null;
    animFrameRef.current = requestAnimationFrame(tick);
  }, [tick]);

  const stopRotation = useCallback(() => {
    isHoveringRef.current = false;
    lastTimeRef.current = null;
    if (animFrameRef.current !== null) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
  }, []);

  useEffect(() => () => stopRotation(), [stopRotation]);

  // Close wheel when clicking anywhere outside the component
  useEffect(() => {
    const handleOutsideClick = (e: MouseEvent) => {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
        setIsOpen(false);
        stopRotation();
        setHoveredModelId(null);
      }
    };
    document.addEventListener("mousedown", handleOutsideClick);
    return () => document.removeEventListener("mousedown", handleOutsideClick);
  }, [stopRotation]);

  const updateBackendModel = async (id: string) => {
    try {
      const token = localStorage.getItem("lexved_token");
      await fetch(`${API_URL}/api/settings/generation_model`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
        body: JSON.stringify({ model: id }),
      });
    } catch (err) {
      console.error(err);
    }
  };

  const handleModelSelect = (id: string) => {
    onModelChange(id);
    updateBackendModel(id);
    // Close the wheel after selection
    setIsOpen(false);
    stopRotation();
    setHoveredModelId(null);
  };

  const handleOrbitMouseMove = (e: React.MouseEvent) => {
    const centerY = window.innerHeight / 2;
    // Upper half (e.clientY < centerY) rotates DOWN (dir = 1)
    // Lower half (e.clientY >= centerY) rotates UP (dir = -1)
    rotateDirRef.current = e.clientY < centerY ? 1 : -1;
  };

  return (
    <aside ref={wrapperRef} className="fixed right-0 top-1/2 -translate-y-1/2 z-40 select-none">
      <div
        className="relative"
        style={{ width: TAB_W, height: TAB_H }}
      >
        {/* ── Orbit ring + model nodes ─────────────────────────── */}
        <AnimatePresence>
          {isOpen && (
            <motion.div
              key="orbit"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8, transition: { duration: 0.25 } }}
              className="absolute"
              style={{ left: TAB_W, top: TAB_H / 2, width: 0, height: 0 }}
              onMouseEnter={() => startRotation()}
              onMouseLeave={() => stopRotation()}
              onMouseMove={handleOrbitMouseMove}
            >
              {/* Invisible outer rotation track surrounding the LLM nodes */}
              <div
                className="absolute rounded-full pointer-events-auto cursor-pointer"
                style={{
                  width: R_OUTER * 2,
                  height: R_OUTER * 2,
                  left: -R_OUTER,
                  top: -R_OUTER,
                  background: "transparent",
                }}
                onMouseEnter={() => startRotation()}
                onMouseLeave={() => stopRotation()}
                onMouseMove={handleOrbitMouseMove}
              />

              {/* Filled dotted orbit guide circle with hover-rotation */}
              <div
                className="absolute rounded-full pointer-events-auto cursor-pointer"
                style={{
                  width: R * 2,
                  height: R * 2,
                  left: -R,
                  top: -R,
                  border: "1.5px dashed rgba(212, 175, 55, 0.35)",
                  background: "rgba(220, 220, 225, 0.08)",
                  backdropFilter: "blur(10px)",
                  boxShadow: "0 0 30px rgba(0,0,0,0.5), inset 0 0 20px rgba(255,255,255,0.05)",
                }}
                onMouseEnter={() => startRotation()}
                onMouseLeave={() => stopRotation()}
                onMouseMove={handleOrbitMouseMove}
              />

              {/* Nodes — evenly spaced at 360/N = 45° apart */}
              {MODEL_LIST.map((model, i) => {
                const isActive = activeModelId === model.id;
                const isHovered = hoveredModelId === model.id;
                // Full circle: 360/N degrees apart, rotating continuously
                const angleDeg = (i * (360 / N)) + rotationDeg;
                const angleRad = (angleDeg * Math.PI) / 180;
                const x = R * Math.cos(angleRad);
                const y = R * Math.sin(angleRad);
                const nodeSize = 80;

                return (
                  <div
                    key={model.id}
                    className="absolute pointer-events-auto"
                    style={{
                      left: x - nodeSize / 2,
                      top: y - nodeSize / 2,
                      width: nodeSize,
                      height: nodeSize,
                    }}
                    onClick={() => handleModelSelect(model.id)}
                    onMouseEnter={() => {
                      stopRotation(); // Pause wheel rotation so node stays stationary under cursor for easy selection!
                      setHoveredModelId(model.id);
                    }}
                    onMouseLeave={() => setHoveredModelId(null)}
                  >
                    {/* Node circle */}
                    <motion.div
                      animate={{ scale: isHovered ? 1.18 : 1 }}
                      transition={{ duration: 0.2 }}
                      className="w-full h-full rounded-full flex flex-col items-center justify-center cursor-pointer relative"
                      style={{
                        background: isActive
                          ? "linear-gradient(135deg, #D4AF37, #b89320)"
                          : isHovered ? "#252525" : "#161616",
                        border: isActive
                          ? "2px solid #FFD700"
                          : isHovered ? "1.5px solid #D4AF37" : "1px solid #2e2e2e",
                        color: isActive ? "#000" : isHovered ? "#fff" : "#888",
                        boxShadow: isActive
                          ? "0 0 24px rgba(212,175,55,0.55)"
                          : isHovered ? "0 0 14px rgba(212,175,55,0.28)" : "none",
                      }}
                    >
                      <span className="material-icons-round text-[26px]">{model.icon}</span>
                      <span
                        className="text-[0.54rem] font-bold tracking-tight uppercase mt-0.5"
                        style={{ color: isActive ? "#000" : "#888" }}
                      >
                        {model.shortName}
                      </span>

                      {/* Active ping ring */}
                      {isActive && (
                        <span className="absolute -inset-1.5 rounded-full border border-[#D4AF37]/40 animate-ping pointer-events-none opacity-50" />
                      )}
                    </motion.div>

                    {/* Hover tooltip */}
                    <AnimatePresence>
                      {isHovered && (
                        <motion.div
                          initial={{ opacity: 0, x: -8, scale: 0.9 }}
                          animate={{ opacity: 1, x: -14, scale: 1 }}
                          exit={{ opacity: 0, scale: 0.9 }}
                          transition={{ duration: 0.18 }}
                          className="absolute right-full mr-2 top-1/2 -translate-y-1/2 px-3 py-1.5 rounded-lg pointer-events-none whitespace-nowrap z-50"
                          style={{
                            background: "#0a0a0a",
                            border: "1px solid rgba(212,175,55,0.5)",
                            boxShadow: "0 4px 20px rgba(0,0,0,0.8), 0 0 10px rgba(212,175,55,0.2)",
                          }}
                        >
                          <p className="text-[0.75rem] font-bold text-white">{model.name}</p>
                          <p className="text-[0.58rem] font-semibold text-[#D4AF37] uppercase tracking-wider">
                            {model.tier}
                          </p>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                );
              })}
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Semicircle trigger tab ─────────────────────────────── */}
        <motion.button
          whileHover={{ scale: 1.06, boxShadow: "-8px 0 30px rgba(212,175,55,0.5)" }}
          whileTap={{ scale: 0.95 }}
          onClick={() => setIsOpen((prev) => !prev)}
          className="absolute right-0 top-0 rounded-l-full flex items-center justify-end pr-0.5 cursor-pointer z-30"
          style={{
            width: TAB_W,
            height: TAB_H,
            background: "linear-gradient(135deg, #1e1e1e, #0a0a0a)",
            borderLeft: "2px solid #D4AF37",
            borderTop: "2px solid #D4AF37",
            borderBottom: "2px solid #D4AF37",
            borderRight: "none",
            boxShadow: isOpen
              ? "-8px 0 30px rgba(212,175,55,0.4), inset 2px 0 12px rgba(212,175,55,0.18)"
              : "-4px 0 18px rgba(212,175,55,0.2)",
            transition: "box-shadow 0.3s",
          }}
          title="LLM Model Selector"
        >
          <motion.div
            animate={{ rotate: isOpen ? 180 : 0 }}
            transition={{ duration: 0.5, ease: "easeInOut" }}
            className="flex items-center justify-center"
          >
            <img
              src="/sparkle.png"
              alt="Sparkle Icon"
              className="w-8 h-8 object-contain drop-shadow-[0_0_6px_rgba(212,175,55,0.4)]"
            />
          </motion.div>
        </motion.button>
      </div>
    </aside>
  );
}
