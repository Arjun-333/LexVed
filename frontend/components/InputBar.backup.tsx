"use client";

import { useState, useRef, useEffect, forwardRef, useImperativeHandle } from "react";

interface InputBarProps {
  onSend: (text: string, agentic: boolean) => void;
  onStop: () => void;
  onUpload?: (file: File) => void;
  disabled?: boolean;
  userInitial?: string;
}

const InputBar = forwardRef<{ focus: () => void }, InputBarProps>(({ onSend, onStop, onUpload, disabled, userInitial = "U" }, ref) => {
  const [text, setText] = useState("");
  const [focused, setFocused] = useState(false);
  const [agentic, setAgentic] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useImperativeHandle(ref, () => ({
    focus: () => inputRef.current?.focus()
  }));

  useEffect(() => { inputRef.current?.focus(); }, []);

  function handleSubmit() {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed, agentic);
    setText("");
  }

  return (
    <div className="w-full max-w-[680px] mx-auto pb-6 pt-2">
      {/* Main pill container */}
      <div
        className="flex items-center gap-3 px-4 py-3 rounded-full transition-all duration-400 relative"
        style={{
          background: "#111111",
          border: focused
            ? "1px solid rgba(212,175,55,0.45)"
            : "1px solid rgba(255,255,255,0.08)",
          boxShadow: focused
            ? "0 0 0 3px rgba(212,175,55,0.08), 0 8px 40px rgba(0,0,0,0.6)"
            : "0 8px 40px rgba(0,0,0,0.5)",
        }}
      >
        {/* Search Icon */}
        <div className="flex items-center justify-center shrink-0" style={{ color: "#444" }}>
          <span className="material-icons-round text-[18px]">search</span>
        </div>

        {/* Attach Icon */}
        <label
          title="Upload PDF"
          className="flex items-center justify-center cursor-pointer shrink-0 transition-colors duration-200"
          style={{ color: focused ? "#666" : "#3a3a3a" }}
        >
          <span className="material-icons-round text-[18px]">attach_file</span>
          <input
            type="file"
            className="hidden"
            accept=".pdf"
            disabled={disabled}
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file && onUpload) {
                onUpload(file);
                e.target.value = "";
              }
            }}
          />
        </label>

        {/* Text Input */}
        <input
          ref={inputRef}
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          placeholder="Ask a legal question..."
          disabled={disabled}
          className="flex-1 bg-transparent border-none outline-none text-[0.9rem]"
          style={{
            color: "#e0e0e0",
            fontWeight: 400,
            letterSpacing: "-0.01em",
          }}
        />

        {/* Avatar Cluster — stacked mini avatars */}
        <div className="avatar-cluster shrink-0">
          <div
            className="av"
            style={{ background: "linear-gradient(135deg, #2a2a2a, #1a1a1a)", color: "#D4AF37" }}
            title="You"
          >
            {userInitial}
          </div>
          <div
            className="av"
            style={{ background: "linear-gradient(135deg, #1a1800, #2a2200)", color: "#D4AF37" }}
            title="LexVed AI"
          >
            L
          </div>
        </div>

        {/* Mode Toggle */}
        <button
          onClick={() => setAgentic(!agentic)}
          disabled={disabled}
          title="Toggle Agent Mode"
          className="flex items-center gap-1.5 shrink-0 rounded-full transition-all duration-300"
          style={{
            padding: "4px 10px",
            background: agentic ? "rgba(212,175,55,0.12)" : "rgba(255,255,255,0.05)",
            border: agentic ? "1px solid rgba(212,175,55,0.35)" : "1px solid rgba(255,255,255,0.07)",
            color: agentic ? "#D4AF37" : "#555555",
            fontSize: "0.65rem",
            fontWeight: 700,
            letterSpacing: "0.06em",
          }}
        >
          <span className="material-icons-round text-[12px]">
            {agentic ? "psychology" : "tune"}
          </span>
          {agentic ? "AGENT" : "STANDARD"}
        </button>

        {/* Send / Stop Button */}
        {disabled ? (
          <button
            onClick={onStop}
            className="w-8 h-8 rounded-full flex items-center justify-center cursor-pointer shrink-0 transition-all duration-300"
            style={{
              background: "rgba(212,175,55,0.12)",
              border: "1px solid rgba(212,175,55,0.3)",
              color: "#D4AF37",
            }}
          >
            <span className="material-icons-round text-[16px]">stop</span>
          </button>
        ) : (
          <button
            onClick={handleSubmit}
            disabled={!text.trim()}
            className="w-8 h-8 rounded-full flex items-center justify-center cursor-pointer shrink-0 transition-all duration-300 hover:scale-110 active:scale-95 disabled:opacity-20 disabled:cursor-not-allowed"
            style={{
              background: text.trim()
                ? "linear-gradient(135deg, #D4AF37, #b8962e)"
                : "rgba(255,255,255,0.05)",
              color: text.trim() ? "#000" : "#555",
              boxShadow: text.trim() ? "0 0 16px rgba(212,175,55,0.3)" : "none",
              border: "none",
            }}
          >
            <span className="material-icons-round text-[16px]">arrow_upward</span>
          </button>
        )}
      </div>

      {/* Footer Status */}
      <div className="flex items-center justify-center gap-2 mt-3">
        <span
          className="w-1.5 h-1.5 rounded-full"
          style={{ background: "#D4AF37", opacity: 0.6 }}
        />
        <p
          className="text-[0.6rem] font-semibold uppercase tracking-[0.12em]"
          style={{ color: "#444444" }}
        >
          Encrypted Local Inference · Multi-Turn Context
        </p>
      </div>
    </div>
  );
});

InputBar.displayName = "InputBar";
export default InputBar;
