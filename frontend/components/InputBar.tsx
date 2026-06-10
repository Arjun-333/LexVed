"use client";

import { useState, useRef, useEffect, forwardRef, useImperativeHandle } from "react";

interface InputBarProps {
  onSend: (text: string, agentic: boolean) => void;
  onStop: () => void;
  onUpload?: (file: File) => void;
  disabled?: boolean;
}

const InputBar = forwardRef<{ focus: () => void }, InputBarProps>(({ onSend, onStop, onUpload, disabled }, ref) => {
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
    <div className="w-full max-w-[700px] mx-auto pb-8 pt-2">
      <div
        className="flex items-center gap-4 px-6 py-4 rounded-[28px] transition-all duration-500"
        style={{
          background: "var(--surface)",
          border: focused ? "1px solid var(--accent)" : "1px solid var(--border)",
          boxShadow: focused ? "var(--shadow-gold)" : "var(--shadow-prestige)",
        }}
      >
        <div className="w-6 h-6 flex items-center justify-center opacity-40">
           <span className="material-icons-round text-[18px]">search</span>
        </div>

        <label 
          title="Upload PDF directly to index"
          className="w-8 h-8 rounded-full flex items-center justify-center cursor-pointer transition-all duration-300 hover:bg-[var(--surface-active)] text-[var(--text-muted)] hover:text-[var(--text)] shrink-0"
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
          className="flex-1 bg-transparent border-none outline-none text-[0.95rem] transition-all duration-300"
          style={{ color: "var(--text)", fontWeight: 400 }}
        />

        {/* Keyboard shortcut hint */}
        {!focused && !text && (
          <div className="hidden md:flex items-center gap-1 opacity-30">
            <kbd className="px-1.5 py-0.5 text-[9px] font-bold border border-[var(--border)] rounded bg-[var(--surface-active)]" style={{ color: "var(--text-muted)" }}>
              Ctrl+K
            </kbd>
          </div>
        )}

        <button 
          onClick={() => setAgentic(!agentic)}
          disabled={disabled}
          title="Toggle LangGraph Agent Mode (AI decides which tools to use)"
          className={`flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-full transition-all text-[10px] font-bold uppercase tracking-widest ${agentic ? "bg-[var(--accent-bg)] border-[var(--accent-dim)] text-[var(--accent)]" : "bg-[var(--surface-active)] border-transparent text-[var(--text-muted)] hover:text-[var(--text)]"} border`}
        >
          <span className="material-icons-round text-[14px]">psychology</span>
          {agentic ? "Agent" : "Standard"}
        </button>

        {disabled ? (
          <button
            onClick={onStop}
            className="w-9 h-9 rounded-xl flex items-center justify-center cursor-pointer
              transition-all duration-400 bg-[var(--accent-bg)] text-[var(--accent)] border border-[var(--accent-dim)]
              hover:bg-[var(--accent)] hover:text-white"
          >
            <span className="material-icons-round text-[18px]">stop</span>
          </button>
        ) : (
          <button
            onClick={handleSubmit}
            disabled={!text.trim()}
            className="w-9 h-9 rounded-full flex items-center justify-center cursor-pointer
              transition-all duration-400 hover:scale-110 active:scale-95
              disabled:opacity-20 disabled:cursor-not-allowed"
            style={{
              background: text.trim() ? "var(--accent)" : "var(--surface-active)",
              color: text.trim() ? "white" : "var(--text-muted)",
              boxShadow: text.trim() ? "var(--shadow-gold)" : "none",
            }}
          >
            <span className="material-icons-round text-[18px]">arrow_upward</span>
          </button>
        )}
      </div>

      <div className="flex items-center justify-center gap-3 mt-4 opacity-40 hover:opacity-100 transition-opacity duration-300">
         <span className="w-1.5 h-1.5 rounded-full" style={{ background: "var(--accent)" }} />
         <p className="text-[0.65rem] font-bold uppercase tracking-[0.1em]" style={{ color: "var(--text-muted)" }}>
           Encrypted Local Inference · Multi-Turn Context
         </p>
      </div>
    </div>
  );
});

InputBar.displayName = "InputBar";
export default InputBar;
