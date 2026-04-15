"use client";

import { useTheme } from "./ThemeProvider";
import { motion } from "framer-motion";
import { useState } from "react";
import MetricsDashboard from "./MetricsDashboard";

export default function Sidebar() {
  const { theme, toggle } = useTheme();
  const [showMetrics, setShowMetrics] = useState(false);

  return (
    <nav
      className="w-[64px] flex flex-col items-center py-6 z-50 transition-all duration-500"
      style={{
        background: "var(--bg-secondary)",
        borderRight: "1px solid var(--border)",
      }}
    >
      {/* Brand mark */}
      <motion.div
        whileHover={{ scale: 1.1, rotate: 5 }}
        className="w-10 h-10 rounded-xl flex items-center justify-center mb-10 cursor-pointer shadow-lg"
        style={{
          background: "linear-gradient(135deg, var(--accent), var(--accent-dim))",
          color: "white",
        }}
        title="LexVed Home"
      >
        <span className="material-icons-round text-[22px]">account_balance</span>
      </motion.div>

      {/* Navigation Items */}
      <div className="flex flex-col gap-6">
        <IconButton active icon="chat_bubble" title="New Brief" />
        <IconButton icon="history" title="Research History" />
        <IconButton icon="folder" title="Case Files" />
      </div>

      <div className="flex-1" />

      {/* Actions */}
      <div className="flex flex-col gap-6 mb-2">
        <IconButton 
          icon="analytics" 
          title="Performance Audit" 
          onClick={() => setShowMetrics(true)} 
        />
        <IconButton
          icon={theme === "dark" ? "light_mode" : "dark_mode"}
          title={theme === "dark" ? "Light Mode" : "Dark Mode"}
          onClick={toggle}
        />
        <IconButton icon="tune" title="System Settings" />
      </div>

      <MetricsDashboard 
        isOpen={showMetrics} 
        onClose={() => setShowMetrics(false)} 
      />
    </nav>
  );
}

function IconButton({ icon, title, active, onClick }: { icon: string; title: string; active?: boolean; onClick?: () => void }) {
  return (
    <motion.button
      onClick={onClick}
      whileHover={{ scale: 1.1 }}
      whileTap={{ scale: 0.95 }}
      className="group relative w-10 h-10 rounded-xl flex items-center justify-center transition-all duration-300"
      style={{
        background: active ? "var(--accent-bg)" : "transparent",
        color: active ? "var(--accent)" : "var(--text-muted)",
        border: active ? "1px solid var(--accent-glow)" : "1px solid transparent",
      }}
      title={title}
    >
      <span className="material-icons-round text-[20px] group-hover:text-[var(--text)] transition-colors">
        {icon}
      </span>

      {/* Tooltip Label (Simple) */}
      <div className="absolute left-14 px-2 py-1 rounded-md bg-black text-white text-[10px] font-bold opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity whitespace-nowrap z-50 uppercase tracking-widest shadow-xl">
        {title}
      </div>
    </motion.button>
  );
}
