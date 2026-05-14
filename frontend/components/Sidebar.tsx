"use client";

import { useTheme } from "./ThemeProvider";
import { useAuth } from "./AuthContext";
import { motion } from "framer-motion";
import { useState, useEffect } from "react";
import MetricsDashboard from "./MetricsDashboard";
import CaseFilesModal from "./CaseFilesModal";
import ResearchHistoryModal from "./ResearchHistoryModal";
import AdminPanel from "./AdminPanel";
import SettingsModal from "./SettingsModal";

export default function Sidebar() {
  const { theme, toggle } = useTheme();
  const { user, isAdmin, logout } = useAuth();
  const [showMetrics, setShowMetrics] = useState(false);
  const [showFiles, setShowFiles] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [showAdmin, setShowAdmin] = useState(false);
  const [showSettings, setShowSettings] = useState(false);

  // Global keyboard shortcuts for modals
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Escape — close any open modal
      if (e.key === "Escape") {
        setShowMetrics(false);
        setShowFiles(false);
        setShowHistory(false);
        setShowAdmin(false);
      }
      // Ctrl+M — Open Metrics Dashboard (admin only)
      if ((e.ctrlKey || e.metaKey) && e.key === "m" && isAdmin) {
        e.preventDefault();
        setShowMetrics(prev => !prev);
      }
      // Ctrl+H — Open Research History
      if ((e.ctrlKey || e.metaKey) && e.key === "h") {
        e.preventDefault();
        setShowHistory(prev => !prev);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isAdmin]);

  return (
    <nav
      className="w-[64px] flex flex-col items-center py-6 pb-14 z-50 transition-all duration-500"
      style={{
        background: "var(--bg-secondary)",
        borderRight: "1px solid var(--border)",
      }}
    >
      {/* Brand mark */}
      <motion.div
        whileHover={{ scale: 1.05 }}
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
        <IconButton 
          icon="history" 
          title="Research History (Ctrl+H)" 
          onClick={() => setShowHistory(true)}
        />
        <IconButton 
          icon="folder" 
          title="Case Files" 
          onClick={() => setShowFiles(true)}
        />
      </div>

      <div className="flex-1" />

      {/* Actions */}
      <div className="flex flex-col gap-6 mb-2">
        {/* Performance Audit — Admin Only */}
        {isAdmin && (
          <IconButton 
            icon="analytics" 
            title="Performance Audit (Ctrl+M)" 
            onClick={() => setShowMetrics(true)} 
          />
        )}

        {/* Admin Console — Admin Only */}
        {isAdmin && (
          <IconButton 
            icon="admin_panel_settings" 
            title="Admin Console" 
            onClick={() => setShowAdmin(true)}
            accent
          />
        )}

        <IconButton
          icon={theme === "dark" ? "light_mode" : "dark_mode"}
          title={theme === "dark" ? "Light Mode" : "Dark Mode"}
          onClick={toggle}
        />
        <IconButton icon="tune" title="System Settings" onClick={() => setShowSettings(true)} />
      </div>

      {/* User Info + Logout */}
      <div className="mt-4 pt-4 flex flex-col items-center gap-3" style={{ borderTop: "1px solid var(--border)" }}>
        {/* User Avatar */}
        <motion.div
          whileHover={{ scale: 1.05 }}
          className="group relative w-10 h-10 rounded-xl flex items-center justify-center cursor-default"
          style={{
            background: isAdmin ? "rgba(212, 175, 55, 0.1)" : "var(--surface)",
            border: isAdmin ? "1px solid rgba(212, 175, 55, 0.2)" : "1px solid var(--border)",
            color: isAdmin ? "var(--accent)" : "var(--text-muted)",
          }}
        >
          <span className="material-icons-round text-[18px]">
            {isAdmin ? "shield" : "person"}
          </span>
          {/* Tooltip with role */}
          <div className="absolute left-14 px-3 py-2 rounded-lg bg-black text-white text-[9px] font-bold opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity whitespace-nowrap z-50 uppercase tracking-widest shadow-xl border border-white/10">
            <div className="text-[#D4AF37] mb-0.5">{user?.displayName}</div>
            <div className="text-white/40">@{user?.username} · {user?.role}</div>
          </div>
        </motion.div>

        {/* Logout */}
        <IconButton 
          icon="logout" 
          title="Sign Out" 
          onClick={logout}
        />
      </div>

      {/* Modals */}
      {isAdmin && (
        <MetricsDashboard 
          isOpen={showMetrics} 
          onClose={() => setShowMetrics(false)} 
        />
      )}
      
      <CaseFilesModal 
        isOpen={showFiles} 
        onClose={() => setShowFiles(false)} 
      />

      <ResearchHistoryModal 
        isOpen={showHistory} 
        onClose={() => setShowHistory(false)} 
      />

      {isAdmin && (
        <AdminPanel 
          isOpen={showAdmin} 
          onClose={() => setShowAdmin(false)} 
        />
      )}

      <SettingsModal
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
      />
    </nav>
  );
}

function IconButton({ icon, title, active, onClick, accent }: { icon: string; title: string; active?: boolean; onClick?: () => void; accent?: boolean }) {
  return (
    <motion.button
      onClick={onClick}
      whileHover={{ scale: 1.1 }}
      whileTap={{ scale: 0.95 }}
      className="group relative w-10 h-10 rounded-xl flex items-center justify-center transition-all duration-300"
      style={{
        background: active ? "var(--accent-bg)" : accent ? "rgba(212, 175, 55, 0.05)" : "transparent",
        color: active ? "var(--accent)" : accent ? "var(--accent)" : "var(--text-muted)",
        border: active ? "1px solid var(--accent-glow)" : accent ? "1px solid rgba(212, 175, 55, 0.15)" : "1px solid transparent",
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
