"use client";

import { useState, useRef, useEffect, forwardRef, useImperativeHandle } from "react";

interface InputBarProps {
  onSend: (text: string, agentic: boolean) => void;
  onStop: () => void;
  onUpload?: (file: File) => void;
  disabled?: boolean;
  userInitial?: string;
}


const InputBar = forwardRef<{ focus: () => void }, InputBarProps>(
  ({ onSend, onStop, onUpload, disabled, userInitial = "U" }, ref) => {
    const [text, setText] = useState("");
    const [agentic, setAgentic] = useState(false);
    const inputRef = useRef<HTMLTextAreaElement>(null);

    useImperativeHandle(ref, () => ({
      focus: () => inputRef.current?.focus(),
    }));

    useEffect(() => {
      inputRef.current?.focus();
    }, []);

    function handleSubmit() {
      const trimmed = text.trim();
      if (!trimmed || disabled) return;
      onSend(trimmed, agentic);
      setText("");
      // Reset textarea height
      if (inputRef.current) inputRef.current.style.height = "auto";
    }

    // Auto-grow textarea
    useEffect(() => {
      const el = inputRef.current;
      if (!el) return;
      el.style.height = "auto";
      el.style.height = `${el.scrollHeight}px`;
    }, [text]);

    return (
      <div className="w-full max-w-[720px] mx-auto pb-6 pt-2">
        {/* ── SINGLE UNIFIED CONTAINER (#121214 color from 2nd image) ── */}
        <div
          className="rounded-[24px] overflow-hidden transition-all duration-300 relative flex flex-col"
          style={{
            background: "#131314",
            border: "1px solid rgba(255,255,255,0.09)",
            boxShadow: "0 16px 48px rgba(0,0,0,0.55)",
          }}
        >
          {/* ── TOP SECTION: TEXTBOX CARD (Flush to Top/Left/Right Borders) ── */}
          <div className="bg-[#131314] px-3.5 pt-2.5 pb-2 rounded-b-[20px] border-b border-white/[0.08] flex flex-col justify-between">
            <style dangerouslySetInnerHTML={{__html: `
              textarea#customInputBarArea::placeholder {
                color: #555555 !important;
                opacity: 1 !important;
              }
            `}} />
            {/* Auto-growing Textarea */}
            <textarea
              id="customInputBarArea"
              ref={inputRef}
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit();
                }
                // Shift+Enter falls through naturally → newline
              }}
              placeholder="Assign a task or type / for more"
              disabled={disabled}
              rows={1}
              className="w-full bg-transparent border-none outline-none text-[0.92rem] py-0.5 tracking-tight resize-none overflow-hidden"
              style={{
                color: "#555555",
                fontWeight: 400,
                minHeight: "24px",
                maxHeight: "160px",
              }}
            />

            {/* Bottom Controls Toolbar */}
            <div className="flex items-center justify-between pt-2 mt-1.5 border-t border-white/[0.04]">
              {/* Left Action Group */}
              <div className="flex items-center gap-2">
                {/* Plus / Upload Button */}
                <label
                  title="Attach PDF Document"
                  className="w-8 h-8 rounded-full bg-white/[0.02] hover:bg-white/[0.06] flex items-center justify-center cursor-pointer transition-colors text-[#555555] hover:text-white"
                >
                  <span className="material-icons-round text-[18px]">add</span>
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

                {/* Mode Toggle Button */}
                <button
                  onClick={() => setAgentic(!agentic)}
                  title="Toggle Agentic Mode"
                  className="w-8 h-8 rounded-full bg-white/[0.03] hover:bg-white/[0.06] flex items-center justify-center cursor-pointer transition-colors text-[#555555] hover:text-white"
                >
                  <span className="material-icons-round text-[16px]">
                    {agentic ? "psychology" : "tune"}
                  </span>
                </button>

                {/* Engine Pill Badge */}
                <button
                  onClick={() => setAgentic(!agentic)}
                  className="flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-white/[0.03] hover:bg-white/[0.06] transition-colors text-[0.78rem] font-medium text-[#555555] hover:text-white"
                >
                  <span className="material-icons-round text-[16px] text-[#555555]/70">
                    laptop
                  </span>
                  <span>{agentic ? "LexVed Agentic" : "LexVed Engine"}</span>
                </button>
              </div>

              {/* Right Action Group */}
              <div className="flex items-center gap-2">
                {/* Speech Waveform Button */}
                <button
                  type="button"
                  className="w-8 h-8 rounded-full hover:bg-white/5 flex items-center justify-center text-[#555555] hover:text-white transition-colors"
                  title="Voice Input"
                >
                  <span className="material-icons-round text-[18px]">graphic_eq</span>
                </button>

                {/* Mic Button */}
                <button
                  type="button"
                  className="w-8 h-8 rounded-full hover:bg-white/5 flex items-center justify-center text-[#555555] hover:text-white transition-colors"
                  title="Microphone"
                >
                  <span className="material-icons-round text-[18px]">mic</span>
                </button>

                {/* Send / Stop Button */}
                {disabled ? (
                  <button
                    onClick={onStop}
                    className="w-8 h-8 rounded-full bg-[#D4AF37]/20 border border-[#D4AF37]/40 text-[#D4AF37] flex items-center justify-center cursor-pointer transition-all"
                    title="Stop generation"
                  >
                    <span className="material-icons-round text-[16px]">stop</span>
                  </button>
                ) : (
                  <button
                    onClick={handleSubmit}
                    disabled={!text.trim()}
                    className="w-8 h-8 rounded-full flex items-center justify-center cursor-pointer transition-all duration-200 disabled:opacity-25 disabled:cursor-not-allowed"
                    style={{
                      background: text.trim() ? "#D4AF37" : "#323237",
                      color: text.trim() ? "#000" : "#777",
                    }}
                    title="Send message"
                  >
                    <span className="material-icons-round text-[18px]">arrow_upward</span>
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }
);

InputBar.displayName = "InputBar";
export default InputBar;
