"use client";

import { useState, useRef, useEffect, forwardRef, useImperativeHandle } from "react";

interface InputBarProps {
  onSend: (text: string) => void;
  disabled?: boolean;
}

const InputBar = forwardRef<{ focus: () => void }, InputBarProps>(({ onSend, disabled }, ref) => {
  const [text, setText] = useState("");
  const [focused, setFocused] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useImperativeHandle(ref, () => ({
    focus: () => inputRef.current?.focus()
  }));

  useEffect(() => { inputRef.current?.focus(); }, []);

  function handleSubmit() {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
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
          onClick={handleSubmit}
          disabled={disabled || !text.trim()}
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
      </div>

      <div className="flex items-center justify-center gap-3 mt-4 opacity-40 hover:opacity-100 transition-opacity duration-300">
         <span className="w-1.5 h-1.5 rounded-full bg-green-500" />
         <p className="text-[0.65rem] font-bold uppercase tracking-[0.1em]" style={{ color: "var(--text-muted)" }}>
           Encrypted Local Inference · Multi-Turn Context
         </p>
      </div>
    </div>
  );
});

InputBar.displayName = "InputBar";
export default InputBar;
