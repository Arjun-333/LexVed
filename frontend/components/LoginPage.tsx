"use client";
import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useAuth } from "./AuthContext";

export default function LoginPage() {
  const { login } = useAuth();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [shake, setShake] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  useEffect(() => { inputRef.current?.focus(); }, []);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!username.trim() || !password.trim()) { setError("Both fields are required"); setShake(true); setTimeout(() => setShake(false), 600); return; }
    setIsLoading(true); setError("");
    const result = await login(username.trim(), password);
    if (!result.success) { setError(result.error || "Invalid credentials"); setIsLoading(false); setShake(true); setTimeout(() => setShake(false), 600); }
  }

  function fillCreds(u: string, p: string) { setUsername(u); setPassword(p); setError(""); }

  return (
    <div className="min-h-screen flex items-center justify-center relative overflow-hidden" style={{ background: "#000" }}>
      <div className="absolute inset-0 pointer-events-none" style={{ background: "radial-gradient(circle at 30% 20%, rgba(212,175,55,0.04) 0%, transparent 50%), radial-gradient(circle at 70% 80%, rgba(212,175,55,0.03) 0%, transparent 50%)" }} />

      <motion.div initial={{ opacity: 0, y: 30, scale: 0.95 }} animate={{ opacity: 1, y: 0, scale: 1 }} transition={{ duration: 0.8, ease: [0.16,1,0.3,1] }} className="relative z-10 w-full max-w-[440px] mx-4">
        {/* Brand */}
        <div className="text-center mb-10">
          <motion.div initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.9, delay: 0.2 }}
            className="w-20 h-20 rounded-2xl flex items-center justify-center mx-auto mb-6"
            style={{ background: "rgba(212,175,55,0.05)", boxShadow: "0 0 30px rgba(212,175,55,0.15), inset 0 0 20px rgba(255,255,255,0.02)", border: "1px solid rgba(212,175,55,0.15)" }}>
            <span className="material-icons-round text-[38px]" style={{ color: "#D4AF37" }}>balance</span>
          </motion.div>
          <h1 className="text-4xl font-bold tracking-tight mb-2" style={{ fontFamily: "'Playfair Display', serif", color: "#F8F8F8" }}>
            Lex<span style={{ color: "#D4AF37" }}>Ved</span>
          </h1>
          <p className="text-sm tracking-wide" style={{ color: "#666", fontFamily: "'Montserrat', sans-serif" }}>Advanced Legal Intelligence Platform</p>
        </div>

        {/* Login Card */}
        <motion.div animate={shake ? { x: [-12,12,-8,8,-4,4,0] } : {}} transition={{ duration: 0.5 }}>
          <form onSubmit={handleSubmit} className="p-8 rounded-2xl relative overflow-hidden" style={{ background: "rgba(10,10,10,0.8)", border: "1px solid rgba(212,175,55,0.15)", boxShadow: "0 4px 50px rgba(0,0,0,0.8), 0 0 0 1px rgba(212,175,55,0.1)", backdropFilter: "blur(20px)" }}>
            <div className="absolute top-0 left-0 right-0 h-[1px]" style={{ background: "linear-gradient(90deg, transparent, rgba(212,175,55,0.4), transparent)" }} />

            <div className="mb-6">
              <h2 className="text-sm font-bold uppercase tracking-[0.2em] mb-1" style={{ color: "#D4AF37" }}>Secure Access</h2>
              <p className="text-[11px]" style={{ color: "#555" }}>Authenticate to access the intelligence node</p>
            </div>

            <div className="mb-5">
              <label className="block text-[10px] font-bold uppercase tracking-[0.15em] mb-2" style={{ color: "#888" }}>Username</label>
              <input ref={inputRef} type="text" value={username} onChange={e => setUsername(e.target.value)} placeholder="Enter username" autoComplete="username"
                className="w-full px-4 py-3 rounded-xl text-sm outline-none transition-all duration-300 focus:border-[#D4AF37] focus:shadow-[0_0_15px_rgba(212,175,55,0.15)]"
                style={{ background: "rgba(20,20,20,0.9)", border: "1px solid rgba(212,175,55,0.15)", color: "#F8F8F8" }} />
            </div>

            <div className="mb-6">
              <label className="block text-[10px] font-bold uppercase tracking-[0.15em] mb-2" style={{ color: "#888" }}>Password</label>
              <input type="password" value={password} onChange={e => setPassword(e.target.value)} placeholder="Enter password" autoComplete="current-password"
                className="w-full px-4 py-3 rounded-xl text-sm outline-none transition-all duration-300 focus:border-[#D4AF37] focus:shadow-[0_0_15px_rgba(212,175,55,0.15)]"
                style={{ background: "rgba(20,20,20,0.9)", border: "1px solid rgba(212,175,55,0.15)", color: "#F8F8F8" }} />
            </div>

            <AnimatePresence>
              {error && (
                <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }}
                  className="flex items-center gap-2 px-4 py-3 rounded-xl mb-4 overflow-hidden" style={{ background: "rgba(220,38,38,0.08)", border: "1px solid rgba(220,38,38,0.2)" }}>
                  <span className="material-icons-round text-[16px]" style={{ color: "#ef4444" }}>error_outline</span>
                  <span className="text-xs font-medium" style={{ color: "#ef4444" }}>{error}</span>
                </motion.div>
              )}
            </AnimatePresence>

            <motion.button type="submit" disabled={isLoading} whileHover={!isLoading ? { scale: 1.02 } : {}} whileTap={!isLoading ? { scale: 0.98 } : {}}
              className="w-full py-3.5 rounded-xl font-bold uppercase tracking-[0.15em] text-xs flex items-center justify-center gap-2 transition-all duration-300 disabled:opacity-50"
              style={{ background: isLoading ? "rgba(212,175,55,0.1)" : "linear-gradient(135deg, #D4AF37, #AA8C2C)", color: isLoading ? "#D4AF37" : "#000", border: "1px solid rgba(212,175,55,0.3)", boxShadow: isLoading ? "none" : "0 0 20px rgba(212,175,55,0.3)" }}>
              {isLoading ? (<><span className="material-icons-round text-[16px] animate-spin">autorenew</span>Authenticating...</>) : (<><span className="material-icons-round text-[16px]">lock_open</span>Authenticate</>)}
            </motion.button>

            <div className="flex items-center gap-3 my-5">
              <div className="flex-1 h-[1px] bg-white/5" /><span className="text-[9px] uppercase tracking-[0.2em] text-white/20 font-bold">Credentials</span><div className="flex-1 h-[1px] bg-white/5" />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <button type="button" onClick={() => fillCreds("user","lexved2025")} className="p-3 rounded-xl text-left transition-all duration-300 hover:border-[#D4AF37]/40 group" style={{ background: "rgba(20,20,20,0.6)", border: "1px solid rgba(255,255,255,0.05)" }}>
                <div className="flex items-center gap-2 mb-1"><span className="material-icons-round text-[14px]" style={{ color: "#666" }}>person</span><span className="text-[10px] font-bold uppercase tracking-[0.1em] group-hover:text-[#D4AF37] transition-colors" style={{ color: "#888" }}>Researcher</span></div>
                <p className="text-[9px] font-mono" style={{ color: "#555" }}>user / lexved2025</p>
              </button>
              <button type="button" onClick={() => fillCreds("admin","lexved@admin")} className="p-3 rounded-xl text-left transition-all duration-300 hover:border-[#D4AF37]/40 group" style={{ background: "rgba(20,20,20,0.6)", border: "1px solid rgba(255,255,255,0.05)" }}>
                <div className="flex items-center gap-2 mb-1"><span className="material-icons-round text-[14px]" style={{ color: "#D4AF37" }}>admin_panel_settings</span><span className="text-[10px] font-bold uppercase tracking-[0.1em] group-hover:text-[#D4AF37] transition-colors" style={{ color: "#888" }}>Administrator</span></div>
                <p className="text-[9px] font-mono" style={{ color: "#555" }}>admin / lexved@admin</p>
              </button>
            </div>
          </form>
        </motion.div>

        <div className="text-center mt-8 flex items-center justify-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-[#D4AF37]/30" />
          <p className="text-[10px] font-bold uppercase tracking-[0.15em]" style={{ color: "#444" }}>Encrypted Local Inference · Zero Cloud Exposure</p>
        </div>
      </motion.div>
    </div>
  );
}
