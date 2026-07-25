"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { useAuth } from "./AuthContext";

interface ModelMeta {
  id: string;
  name: string;
  icon: string;
  tier: string;
}

const FALLBACK_MODELS: ModelMeta[] = [
  { id: "mixtral-8x7b-32768", name: "Mixtral 8x7B", icon: "all_inclusive", tier: "cloud" },
  { id: "ensemble", name: "Ensemble Logic", icon: "hub", tier: "orchestrator" },
  { id: "llama3", name: "Local Llama 3 8B", icon: "memory", tier: "local" },
  { id: "llama-3.3-70b-versatile", name: "Llama 3.3 70B", icon: "auto_awesome", tier: "cloud" },
  { id: "llama-3.1-8b-instant", name: "Llama 3.1 8B", icon: "bolt", tier: "cloud" },
  { id: "qwen2.5:7b", name: "Local Qwen 2.5", icon: "security", tier: "local" },
];

export default function ModelWheel({
  activeModelId,
  onModelChange,
}: {
  activeModelId: string;
  onModelChange: (id: string) => void;
}) {
  const { authFetch } = useAuth();
  const [modelsList, setModelsList] = useState<ModelMeta[]>(FALLBACK_MODELS);
  const [healthStatus, setHealthStatus] = useState("OFFLINE");
  const [isExpanded, setIsExpanded] = useState(false);

  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:5000";

  useEffect(() => {
    authFetch(`${API_URL}/api/settings/config`)
      .then(res => res.json())
      .then(data => {
        if (data.generation_model_metadata && data.generation_model_metadata.length > 0) {
          setModelsList(data.generation_model_metadata);
        }
      })
      .catch(() => {});
  }, [authFetch, API_URL]);

  useEffect(() => {
    let isMounted = true;
    const checkHealth = () => {
      authFetch(`${API_URL}/api/health`)
        .then(res => res.json())
        .then(data => {
          if (!isMounted) return;
          if (data.ollama === "connected" || data.vector_db === "connected") setHealthStatus("OPTIMAL");
          else setHealthStatus("OFFLINE");
        })
        .catch(() => { if (isMounted) setHealthStatus("OFFLINE"); });
    };
    checkHealth();
    const interval = setInterval(checkHealth, 20000);
    return () => { isMounted = false; clearInterval(interval); };
  }, [authFetch, API_URL]);

  const updateBackendModel = async (id: string) => {
    try {
      const token = localStorage.getItem("lexved_token");
      await fetch(`${API_URL}/api/settings/generation_model`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "Authorization": `Bearer ${token}` },
        body: JSON.stringify({ model: id }),
      });
    } catch (err) {
      console.error(err);
    }
  };

  const handleModelClick = (id: string) => {
    onModelChange(id);
    updateBackendModel(id);
  };

  // Show top 3 models in the primary stack (ensemble in middle = index 1)
  const primaryModels = modelsList.slice(0, 3);
  const extraModels = modelsList.slice(3);

  const statusColor =
    healthStatus === "OPTIMAL" ? "#D4AF37" :
    healthStatus === "DEGRADED" ? "#FFA500" : "#FF6B6B";

  const statusLabel =
    healthStatus === "OPTIMAL" ? "OPTIMAL" :
    healthStatus === "DEGRADED" ? "DEGRADED" : "OFFLINE";

  const getModelIcon = (model: ModelMeta) => {
    // Return a more fitting icon per tier
    if (model.tier === "orchestrator") return "hub";
    if (model.tier === "local") return "memory";
    return model.icon || "auto_awesome";
  };

  return (
    <aside
      className="w-[280px] hidden xl:flex flex-col items-center justify-center py-8 px-6 z-40 select-none shrink-0 relative"
      style={{
        background: "transparent",
        borderLeft: "1px solid #1f1f1f",
      }}
    >
      {/* Intelligence Engine Header */}
      <div className="text-center mb-8 w-full">
        <p
          className="text-[0.6rem] font-bold uppercase tracking-[0.2em] mb-1"
          style={{ color: "#D4AF37" }}
        >
          Intelligence Engine
        </p>
        <p className="text-[0.58rem] font-semibold uppercase tracking-[0.08em]" style={{ color: "#444444" }}>
          LOCAL CLUSTER STATUS:{" "}
          <span style={{ color: statusColor }}>{statusLabel}</span>
        </p>
      </div>

      {/* Model Stack Cards */}
      <div className="flex flex-col gap-3 w-[210px]">
        {primaryModels.map((model, i) => {
          const isActive = activeModelId === model.id;
          const iconName = getModelIcon(model);

          return (
            <motion.div
              key={model.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: i * 0.07, ease: [0.16, 1, 0.3, 1] }}
              onClick={() => handleModelClick(model.id)}
              className={`model-status-card w-full ${isActive ? "active" : ""}`}
            >
              {/* Icon */}
              <span
                className="material-icons-round text-[24px] mb-1"
                style={{
                  color: isActive ? "#D4AF37" : "#3a3a3a",
                  filter: isActive ? "drop-shadow(0 0 8px rgba(212,175,55,0.5))" : "none",
                  transition: "color 0.3s, filter 0.3s",
                }}
              >
                {iconName}
              </span>

              {/* Model Name */}
              <span
                className="font-bold text-[0.82rem] text-center leading-tight"
                style={{
                  color: isActive ? "#D4AF37" : "#cccccc",
                  letterSpacing: "-0.01em",
                  transition: "color 0.3s",
                }}
              >
                {model.name}
              </span>

              {/* Status Badge */}
              <span
                className="text-[0.52rem] font-bold uppercase tracking-[0.2em] mt-0.5"
                style={{
                  color: isActive ? "rgba(212,175,55,0.7)" : "#3a3a3a",
                  transition: "color 0.3s",
                }}
              >
                {isActive ? "ACTIVE LINK" : "STANDBY"}
              </span>
            </motion.div>
          );
        })}

        {/* Extra models (collapsed by default) */}
        {extraModels.length > 0 && (
          <>
            <button
              onClick={() => setIsExpanded(prev => !prev)}
              className="w-full flex items-center justify-center gap-2 py-2 rounded-lg transition-all duration-200"
              style={{
                background: "rgba(255,255,255,0.03)",
                border: "1px solid #222",
                color: "#444",
                fontSize: "0.65rem",
                fontWeight: 600,
                letterSpacing: "0.08em",
              }}
            >
              <span className="material-icons-round text-[14px]">
                {isExpanded ? "expand_less" : "expand_more"}
              </span>
              {isExpanded ? "HIDE" : `+${extraModels.length} MORE`}
            </button>

            {isExpanded && extraModels.map((model, i) => {
              const isActive = activeModelId === model.id;
              return (
                <motion.div
                  key={model.id}
                  initial={{ opacity: 0, y: -8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: i * 0.05 }}
                  onClick={() => handleModelClick(model.id)}
                  className={`model-status-card w-full ${isActive ? "active" : ""}`}
                >
                  <span
                    className="material-icons-round text-[20px]"
                    style={{ color: isActive ? "#D4AF37" : "#333" }}
                  >
                    {getModelIcon(model)}
                  </span>
                  <span className="font-bold text-[0.78rem] text-center" style={{ color: isActive ? "#D4AF37" : "#aaa" }}>
                    {model.name}
                  </span>
                  <span className="text-[0.5rem] font-bold uppercase tracking-[0.18em]" style={{ color: isActive ? "rgba(212,175,55,0.6)" : "#333" }}>
                    {isActive ? "ACTIVE LINK" : "STANDBY"}
                  </span>
                </motion.div>
              );
            })}
          </>
        )}
      </div>

      {/* Up/Down Navigation Arrows */}
      <div className="flex gap-3 mt-8">
        <button
          onClick={() => {
            const idx = modelsList.findIndex(m => m.id === activeModelId);
            const prev = modelsList[(idx - 1 + modelsList.length) % modelsList.length];
            handleModelClick(prev.id);
          }}
          className="w-9 h-9 rounded-full flex items-center justify-center transition-all duration-200 border"
          style={{
            background: "rgba(255,255,255,0.03)",
            borderColor: "#222",
            color: "#555",
          }}
          title="Previous model"
        >
          <span className="material-icons-round text-[18px]">keyboard_arrow_up</span>
        </button>
        <button
          onClick={() => {
            const idx = modelsList.findIndex(m => m.id === activeModelId);
            const next = modelsList[(idx + 1) % modelsList.length];
            handleModelClick(next.id);
          }}
          className="w-9 h-9 rounded-full flex items-center justify-center transition-all duration-200 border"
          style={{
            background: "rgba(255,255,255,0.03)",
            borderColor: "#222",
            color: "#555",
          }}
          title="Next model"
        >
          <span className="material-icons-round text-[18px]">keyboard_arrow_down</span>
        </button>
      </div>
    </aside>
  );
}
