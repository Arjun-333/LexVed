"use client";

import { useAuth } from "./AuthContext";
import { motion, AnimatePresence } from "framer-motion";
import { useState, useEffect } from "react";
import MetricsDashboard from "./MetricsDashboard";
import CaseFilesModal from "./CaseFilesModal";
import ResearchHistoryModal from "./ResearchHistoryModal";
import AdminPanel from "./AdminPanel";
import SettingsModal from "./SettingsModal";

interface SidebarProps {
  onNewBrief?: () => void;
  activeSection?: string;
  onSectionChange?: (section: string) => void;
}

export default function Sidebar({ onNewBrief, activeSection = "Criminal Cases", onSectionChange }: SidebarProps) {
  const { user, isAdmin, logout } = useAuth();
  const [showMetrics, setShowMetrics] = useState(false);
  const [showFiles, setShowFiles] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [showAdmin, setShowAdmin] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [hoveredItem, setHoveredItem] = useState<string | null>(null);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setShowMetrics(false);
        setShowFiles(false);
        setShowHistory(false);
        setShowAdmin(false);
        setShowSettings(false);
      }
      if ((e.ctrlKey || e.metaKey) && e.key === "m" && isAdmin) {
        e.preventDefault();
        setShowMetrics(prev => !prev);
      }
      if ((e.ctrlKey || e.metaKey) && e.key === "h") {
        e.preventDefault();
        setShowHistory(prev => !prev);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isAdmin]);

  const displayName = user?.displayName || user?.username || "Counsel";
  const userInitial = displayName.charAt(0).toUpperCase();

  const portfolioItems = [
    { id: "Onboarding", label: "LexVed Onboarding getting started", icon: "folder", indent: true, onClick: onNewBrief },
    { id: "Briefs", label: "Briefs", icon: "folder", indent: true, onClick: onNewBrief },
    { id: "Criminal Cases", label: "Criminal Cases", icon: "folder", indent: true, active: true, onClick: () => onSectionChange?.("Criminal Cases") },
    { id: "Civil Litigation", label: "Civil Litigation", icon: "folder", indent: true, onClick: () => onSectionChange?.("Civil Litigation") },
  ];

  return (
    <nav
      className="w-[240px] shrink-0 flex flex-col h-full overflow-hidden z-50"
      style={{
        background: "#111111",
        borderRight: "1px solid #1f1f1f",
      }}
    >
      {/* Brand Header */}
      <div className="px-4 pt-5 pb-3 flex items-center gap-3 shrink-0">
        <motion.div
          whileHover={{ scale: 1.08 }}
          whileTap={{ scale: 0.95 }}
          onClick={onNewBrief}
          className="w-8 h-8 rounded-lg flex items-center justify-center cursor-pointer shrink-0"
          style={{
            background: "linear-gradient(135deg, #f5c518, #d4a017)",
            color: "#000",
            boxShadow: "0 2px 12px rgba(245, 197, 24, 0.25)",
          }}
          title="LexVed Home"
        >
          <span className="material-icons-round text-[17px]">account_balance</span>
        </motion.div>
        <span
          className="font-bold text-[1.05rem] tracking-tight"
          style={{ color: "#ffffff", fontFamily: "var(--font-serif)", letterSpacing: "-0.02em" }}
        >
          Lex<span style={{ color: "#f5c518" }}>Ved</span>
        </span>
      </div>

      {/* Inbox */}
      <div className="px-3 mt-1 mb-3 shrink-0">
        <TreeItem
          icon="inbox"
          label="Inbox"
          hovered={hoveredItem === "inbox"}
          onHover={() => setHoveredItem("inbox")}
          onLeave={() => setHoveredItem(null)}
          onClick={onNewBrief}
        />
      </div>

      {/* Legal Portfolios Section */}
      <div className="px-3 flex-1 overflow-y-auto overflow-x-hidden">
        <p
          className="text-[0.63rem] font-semibold uppercase tracking-[0.1em] px-3 mb-2 mt-1"
          style={{ color: "#555555", letterSpacing: "0.08em" }}
        >
          Legal Portfolios
        </p>

        <div className="flex flex-col gap-0.5">
          {portfolioItems.map((item) => (
            <TreeItem
              key={item.id}
              icon={item.icon}
              label={item.label}
              active={activeSection === item.id}
              hovered={hoveredItem === item.id}
              onHover={() => setHoveredItem(item.id)}
              onLeave={() => setHoveredItem(null)}
              onClick={item.onClick}
              indent={item.indent}
            />
          ))}

          {/* Divider */}
          <div className="my-2" style={{ borderTop: "1px solid #1f1f1f" }} />

          <TreeItem
            icon="list_alt"
            label="Case Files"
            hovered={hoveredItem === "casefiles"}
            onHover={() => setHoveredItem("casefiles")}
            onLeave={() => setHoveredItem(null)}
            onClick={() => setShowFiles(true)}
            iconStyle="format_list_bulleted"
          />

          <TreeItem
            icon="tag"
            label="Browse all Portfolios"
            hovered={hoveredItem === "browse"}
            onHover={() => setHoveredItem("browse")}
            onLeave={() => setHoveredItem(null)}
            onClick={() => setShowHistory(true)}
            iconStyle="apps"
          />

          <TreeItem
            icon="add"
            label="Create new portfolio"
            hovered={hoveredItem === "create"}
            onHover={() => setHoveredItem("create")}
            onLeave={() => setHoveredItem(null)}
            onClick={onNewBrief}
            iconStyle="add"
            muted
          />
        </div>

        <div className="flex-1" />
      </div>

      {/* Bottom Section */}
      <div className="px-3 mt-2 mb-2 shrink-0">
        <div className="flex flex-col gap-0.5">
          {isAdmin && (
            <TreeItem
              icon="analytics"
              label="Analytics"
              hovered={hoveredItem === "analytics"}
              onHover={() => setHoveredItem("analytics")}
              onLeave={() => setHoveredItem(null)}
              onClick={() => setShowMetrics(true)}
            />
          )}
          {isAdmin && (
            <TreeItem
              icon="admin_panel_settings"
              label="Admin Panel"
              hovered={hoveredItem === "admin"}
              onHover={() => setHoveredItem("admin")}
              onLeave={() => setHoveredItem(null)}
              onClick={() => setShowAdmin(true)}
              accent
            />
          )}
          <TreeItem
            icon="manage_accounts"
            label="Brand Profile"
            hovered={hoveredItem === "brand"}
            onHover={() => setHoveredItem("brand")}
            onLeave={() => setHoveredItem(null)}
            onClick={() => setShowSettings(true)}
            iconStyle="person_outline"
          />
          <TreeItem
            icon="help_outline"
            label="Support"
            hovered={hoveredItem === "support"}
            onHover={() => setHoveredItem("support")}
            onLeave={() => setHoveredItem(null)}
            onClick={() => {}}
            iconStyle="help_outline"
          />
        </div>
      </div>

      {/* User Row */}
      <div
        className="px-4 py-3 flex items-center justify-between shrink-0"
        style={{ borderTop: "1px solid #1f1f1f" }}
      >
        <div className="flex items-center gap-2.5 min-w-0">
          {/* Avatar circle with gradient */}
          <div
            className="w-8 h-8 rounded-full flex items-center justify-center shrink-0 text-[12px] font-bold"
            style={{
              background: "linear-gradient(135deg, #2a2a2a, #3a3a3a)",
              border: "1.5px solid #333",
              color: "#D4AF37",
              boxShadow: "0 0 0 2px rgba(212,175,55,0.15)",
            }}
          >
            {userInitial}
          </div>
          <div className="min-w-0">
            <p className="text-[0.78rem] font-semibold truncate" style={{ color: "#e0e0e0", letterSpacing: "-0.01em" }}>
              {displayName}
            </p>
          </div>
        </div>
        <button
          onClick={logout}
          className="w-7 h-7 rounded-md flex items-center justify-center transition-all duration-200 hover:bg-[rgba(255,255,255,0.06)]"
          style={{ color: "#555555" }}
          title="Sign Out"
        >
          <span className="material-icons-round text-[15px]">logout</span>
        </button>
      </div>

      {/* Modals */}
      {isAdmin && <MetricsDashboard isOpen={showMetrics} onClose={() => setShowMetrics(false)} />}
      <CaseFilesModal isOpen={showFiles} onClose={() => setShowFiles(false)} />
      <ResearchHistoryModal isOpen={showHistory} onClose={() => setShowHistory(false)} />
      {isAdmin && <AdminPanel isOpen={showAdmin} onClose={() => setShowAdmin(false)} />}
      <SettingsModal isOpen={showSettings} onClose={() => setShowSettings(false)} />
    </nav>
  );
}

function TreeItem({
  icon,
  label,
  active,
  hovered,
  onHover,
  onLeave,
  onClick,
  indent,
  accent,
  muted,
  iconStyle,
}: {
  icon: string;
  label: string;
  active?: boolean;
  hovered?: boolean;
  onHover: () => void;
  onLeave: () => void;
  onClick?: () => void;
  indent?: boolean;
  accent?: boolean;
  muted?: boolean;
  iconStyle?: string;
}) {
  const resolvedIcon = iconStyle || icon;

  return (
    <button
      onClick={onClick}
      onMouseEnter={onHover}
      onMouseLeave={onLeave}
      className="w-full text-left flex items-center gap-2 px-3 py-[5px] rounded-[6px] transition-all duration-150 relative overflow-hidden group"
      style={{
        background: active
          ? "rgba(212,175,55,0.10)"
          : hovered
          ? "rgba(255,255,255,0.04)"
          : "transparent",
        color: active
          ? "#ffffff"
          : accent
          ? "#D4AF37"
          : muted
          ? "#555555"
          : "#7a7a7a",
        paddingLeft: indent ? "20px" : "12px",
      }}
      title={label}
    >
      {/* Active indicator bar */}
      {active && (
        <motion.div
          layoutId="activeBar"
          className="absolute left-0 top-1 bottom-1 w-[2.5px] rounded-r-full"
          style={{ background: "#D4AF37" }}
        />
      )}

      {/* Icon */}
      <span
        className="material-icons-round text-[15px] shrink-0 transition-colors"
        style={{
          color: active ? "#D4AF37" : accent ? "#D4AF37" : muted ? "#555555" : hovered ? "#aaaaaa" : "#555555",
          fontSize: resolvedIcon === "add" ? "16px" : "15px",
        }}
      >
        {resolvedIcon}
      </span>

      {/* Label */}
      <span
        className="text-[0.8rem] font-medium truncate transition-colors"
        style={{
          color: active ? "#f0f0f0" : accent ? "#D4AF37" : muted ? "#555555" : hovered ? "#cccccc" : "#777777",
          letterSpacing: "-0.01em",
        }}
      >
        {label}
      </span>
    </button>
  );
}
