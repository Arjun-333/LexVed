"use client";
import { useState, useRef, useEffect } from "react";
import { motion } from "framer-motion";
import { useAuth } from "./AuthContext";

export default function LoginPage() {
  const { login } = useAuth();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!username.trim() || !password.trim()) {
      setError("Both username and password are required");
      return;
    }
    setIsLoading(true);
    setError("");
    const result = await login(username.trim(), password);
    if (!result.success) {
      setError(result.error || "Invalid credentials");
      setIsLoading(false);
    }
  }

  function fillCreds(u: string, p: string) {
    setUsername(u);
    setPassword(p);
    setError("");
  }

  return (
    <div
      className="min-h-screen flex flex-col justify-between relative overflow-hidden select-none"
      style={{
        background: "#111110",
        fontFamily: "'Montserrat', sans-serif",
        color: "#FFFFFF",
      }}
    >
      {/* ── Background Floating Ambient Icon Pattern (Matches Interface) ───── */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden z-0 select-none">
        {[
          // Top Row
          { src: "/gavel.png", top: "5%", left: "3%", width: 55, duration: 5.2, delay: 0 },
          { src: "/id.png", top: "8%", left: "15%", width: 60, duration: 6.8, delay: 0.4 },
          { src: "/court.png", top: "4%", left: "30%", width: 70, duration: 7.1, delay: 0.9 },
          { src: "/weight.png", top: "7%", left: "45%", width: 50, duration: 5.9, delay: 1.3 },
          { src: "/id.png", top: "4%", right: "32%", width: 58, duration: 6.3, delay: 0.6 },
          { src: "/weight.png", top: "6%", right: "18%", width: 62, duration: 4.8, delay: 0.8 },
          { src: "/court.png", top: "8%", right: "5%", width: 75, duration: 7.2, delay: 0.2 },
          
          // Upper Mid Row
          { src: "/court.png", top: "25%", left: "6%", width: 70, duration: 6.0, delay: 1.0 },
          { src: "/weight.png", top: "28%", left: "20%", width: 52, duration: 5.5, delay: 0.6 },
          { src: "/gavel.png", top: "22%", left: "36%", width: 48, duration: 4.9, delay: 0.3 },
          { src: "/id.png", top: "24%", right: "35%", width: 54, duration: 6.7, delay: 1.1 },
          { src: "/id.png", top: "26%", right: "16%", width: 58, duration: 6.4, delay: 1.2 },
          { src: "/gavel.png", top: "22%", right: "4%", width: 65, duration: 5.0, delay: 0.3 },

          // Lower Mid Row
          { src: "/gavel.png", top: "48%", left: "4%", width: 62, duration: 5.4, delay: 0.7 },
          { src: "/id.png", top: "46%", left: "16%", width: 50, duration: 6.1, delay: 0.2 },
          { src: "/court.png", top: "52%", left: "32%", width: 65, duration: 7.4, delay: 1.4 },
          { src: "/weight.png", top: "48%", right: "30%", width: 56, duration: 5.8, delay: 0.8 },
          { src: "/court.png", top: "50%", right: "14%", width: 68, duration: 6.6, delay: 0.5 },
          { src: "/weight.png", top: "45%", right: "3%", width: 54, duration: 5.1, delay: 1.0 },

          // Bottom Row
          { src: "/id.png", bottom: "16%", left: "5%", width: 65, duration: 5.6, delay: 0.7 },
          { src: "/court.png", bottom: "8%", left: "14%", width: 75, duration: 6.9, delay: 0.1 },
          { src: "/gavel.png", bottom: "18%", left: "26%", width: 52, duration: 4.7, delay: 0.9 },
          { src: "/weight.png", bottom: "6%", left: "40%", width: 58, duration: 6.2, delay: 1.2 },
          { src: "/id.png", bottom: "18%", right: "38%", width: 50, duration: 5.5, delay: 0.3 },
          { src: "/court.png", bottom: "10%", right: "24%", width: 70, duration: 7.0, delay: 0.5 },
          { src: "/gavel.png", bottom: "15%", right: "12%", width: 68, duration: 5.3, delay: 0.9 },
          { src: "/weight.png", bottom: "6%", right: "4%", width: 64, duration: 6.2, delay: 0.4 },
        ].map((item, idx) => (
          <motion.img
            key={idx}
            src={item.src}
            alt="Floating icon"
            animate={{
              y: idx % 2 === 0 ? [-8, 8, -8] : [8, -8, 8],
              rotate: idx % 3 === 0 ? [-5, 5, -5] : [5, -5, 5],
            }}
            transition={{
              duration: item.duration,
              repeat: Infinity,
              ease: "easeInOut",
              delay: item.delay,
            }}
            style={{
              position: "absolute",
              top: item.top,
              bottom: item.bottom,
              left: item.left,
              right: item.right,
              width: `${item.width}px`,
            }}
            className="opacity-[0.11] drop-shadow-sm"
          />
        ))}
      </div>

      {/* ── Top Header Navigation ─────────────────────────────────────────── */}
      <header className="w-full px-8 pt-6 flex items-center justify-between z-20">
        <div className="flex items-center gap-3">
          <img src="/lexvedLogo.png" alt="LexVed Logo" className="w-9 h-9 object-contain" />
          <h1 className="text-3xl font-extrabold tracking-tight flex items-center" style={{ fontFamily: "'Montserrat', sans-serif" }}>
            <span style={{ color: "#FFFFFF" }}>Lex</span>
            <span style={{ color: "#FFD700" }}>Ved</span>
          </h1>
        </div>

        <div>
          <button
            onClick={() => fillCreds("admin", "lexved@admin")}
            className="px-6 py-2.5 rounded-full text-xs font-bold uppercase tracking-wider transition-transform hover:scale-105 cursor-pointer"
            style={{ background: "#FFD700", color: "#111110" }}
          >
            Login
          </button>
        </div>
      </header>

      {/* ── Main Canvas with Stationary Center Card & Direct Sub-Footer ───── */}
      <main className="flex-1 flex items-center justify-center relative px-6 py-8 z-10">
        <div className="relative z-10 w-full max-w-[440px] flex flex-col items-center">
          <div
            className="w-full px-6 py-5 rounded-[24px] relative overflow-hidden"
            style={{
              background: "#181816",
              border: "2px solid rgba(255, 215, 0, 0.55)",
              fontFamily: "'Montserrat', sans-serif",
            }}
          >
            {/* Top golden accent line */}
            <div
              className="absolute top-0 left-0 right-0 h-[2px]"
              style={{ background: "linear-gradient(90deg, transparent, #FFD700, transparent)" }}
            />

            {/* Header text inside card */}
            <div className="text-center mb-5">
              <h2
                style={{
                  fontFamily: "'Montserrat', sans-serif",
                  fontWeight: 800,
                  fontSize: "1.6rem",
                  letterSpacing: "-0.02em",
                  color: "#FFFFFF",
                  marginBottom: "0",
                }}
              >
                Sign in
              </h2>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
              {/* Username Input */}
              <div>
                <label
                  style={{
                    display: "block",
                    fontFamily: "'Montserrat', sans-serif",
                    fontWeight: 700,
                    fontSize: "0.65rem",
                    letterSpacing: "0.18em",
                    textTransform: "uppercase",
                    color: "#C4A35A",
                    marginBottom: "0.4rem",
                  }}
                >
                  Username
                </label>
                <input
                  ref={inputRef}
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder="Enter username"
                  autoComplete="username"
                  className="w-full px-4 py-3 rounded-xl text-xs font-medium outline-none transition-all duration-200 placeholder:text-[#555]"
                  style={{
                    background: "#111110",
                    border: "1px solid #2A2A28",
                    color: "#FFFFFF",
                    fontFamily: "'Montserrat', sans-serif",
                  }}
                />
              </div>

              {/* Password Input with Hide/Show Toggle */}
              <div className="relative">
                <label
                  style={{
                    display: "block",
                    fontFamily: "'Montserrat', sans-serif",
                    fontWeight: 700,
                    fontSize: "0.65rem",
                    letterSpacing: "0.18em",
                    textTransform: "uppercase",
                    color: "#C4A35A",
                    marginBottom: "0.4rem",
                  }}
                >
                  Password
                </label>
                <div className="relative">
                  <input
                    type={showPassword ? "text" : "password"}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="Enter password"
                    autoComplete="current-password"
                    className="w-full px-4 py-3 rounded-xl text-xs font-medium outline-none transition-all duration-200 placeholder:text-[#555] pr-16"
                    style={{
                      background: "#111110",
                      border: "1px solid #2A2A28",
                      color: "#FFFFFF",
                      fontFamily: "'Montserrat', sans-serif",
                    }}
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-4 top-1/2 -translate-y-1/2 text-[11px] font-semibold tracking-wider uppercase transition-colors cursor-pointer"
                    style={{ color: "#C4A35A", fontFamily: "'Montserrat', sans-serif" }}
                  >
                    {showPassword ? "Hide" : "Show"}
                  </button>
                </div>
              </div>

              {/* Trouble link */}
              <div className="flex justify-start">
                <button
                  type="button"
                  onClick={() => fillCreds("user", "lexved2025")}
                  className="text-[11px] font-medium transition-colors hover:underline cursor-pointer"
                  style={{ color: "#8A8070" }}
                >
                  Having trouble in sign in?
                </button>
              </div>

              {/* Error message display */}
              {error && (
                <div
                  className="flex items-center gap-2 px-4 py-2.5 rounded-xl text-xs font-medium overflow-hidden"
                  style={{ background: "rgba(220, 38, 38, 0.12)", border: "1px solid rgba(220, 38, 38, 0.3)", color: "#FF6B6B" }}
                >
                  <span className="material-icons-round text-[16px]">error_outline</span>
                  {error}
                </div>
              )}

              {/* Main Submit Button */}
              <button
                type="submit"
                disabled={isLoading}
                className="w-full py-3.5 rounded-2xl font-bold uppercase tracking-wider text-xs flex items-center justify-center gap-2 transition-all mt-2 cursor-pointer disabled:opacity-50"
                style={{
                  background: "#FFD700",
                  color: "#111110",
                  fontFamily: "'Montserrat', sans-serif",
                }}
              >
                {isLoading ? (
                  <>
                    <span className="material-icons-round text-[16px] animate-spin">autorenew</span>
                    Signing in...
                  </>
                ) : (
                  "Sign in"
                )}
              </button>
            </form>

            {/* Divider */}
            <div className="flex items-center gap-3 my-5">
              <div className="flex-1 h-[1px]" style={{ background: "#2A2A28" }} />
              <span className="text-[10px] uppercase tracking-widest font-semibold" style={{ color: "#666" }}>
                — Or Sign in with —
              </span>
              <div className="flex-1 h-[1px]" style={{ background: "#2A2A28" }} />
            </div>

            {/* Social / Quick credentials buttons */}
            <div className="grid grid-cols-2 gap-3">
              <button
                type="button"
                onClick={() => fillCreds("user", "lexved2025")}
                className="py-2.5 px-3 rounded-xl flex items-center justify-center gap-2 transition-all hover:border-[#FFD700]/40 text-xs font-medium cursor-pointer"
                style={{ background: "#111110", border: "1px solid #2A2A28", color: "#FFFFFF" }}
              >
                <span className="material-icons-round text-[14px]" style={{ color: "#C4A35A" }}>person</span>
                Researcher
              </button>
              <button
                type="button"
                onClick={() => fillCreds("admin", "lexved@admin")}
                className="py-2.5 px-3 rounded-xl flex items-center justify-center gap-2 transition-all hover:border-[#FFD700]/40 text-xs font-medium cursor-pointer"
                style={{ background: "#111110", border: "1px solid #2A2A28", color: "#FFFFFF" }}
              >
                <span className="material-icons-round text-[14px]" style={{ color: "#FFD700" }}>admin_panel_settings</span>
                Administrator
              </button>
            </div>

            {/* Request Account link */}
            <div className="text-center mt-5">
              <p className="text-[11px] font-medium" style={{ color: "#8A8070" }}>
                Don't have an account?{" "}
                <button
                  type="button"
                  onClick={() => fillCreds("user", "lexved2025")}
                  className="font-bold transition-colors hover:underline cursor-pointer"
                  style={{ color: "#FFD700" }}
                >
                  Request Now
                </button>
              </p>
            </div>
          </div>

          {/* Copyright text placed directly below the container */}
          <div className="mt-5 text-center text-[11px] font-medium" style={{ color: "#666" }}>
            Copyright @LexVed 2026 &nbsp;|&nbsp; Privacy Policy
          </div>
        </div>
      </main>
    </div>
  );
}

