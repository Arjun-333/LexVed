"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface LandingPageProps {
  onEnter: () => void;
}

export default function LandingPage({ onEnter }: LandingPageProps) {
  const [visible, setVisible] = useState(true);

  function handleClick() {
    setVisible(false);
    setTimeout(onEnter, 700);
  }

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          key="landing"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0, scale: 1.03 }}
          transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
          className="fixed inset-0 z-[9999] flex items-center justify-center select-none"
          style={{
            background: "#111110",
            overflow: "hidden",
            scrollbarWidth: "none",
          }}
        >

          {/* Floating background graphics pattern (multiplied density) */}
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
              { src: "/weight.png", bottom: "5%", right: "4%", width: 64, duration: 6.2, delay: 0.4 },
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

          {/* Center content */}
          <div className="relative z-10 flex flex-col items-center text-center px-8">

            {/* Eyebrow label */}
            <p
              style={{
                fontFamily: "'Poppins', sans-serif",
                fontWeight: 600,
                fontSize: "0.7rem",
                letterSpacing: "0.22em",
                textTransform: "uppercase",
                color: "#C4A35A",
                marginBottom: "1.4rem",
              }}
            >
              Advanced Legal Intelligence
            </p>

            {/* Main headline */}
            <h1
              style={{
                fontFamily: "'Poppins', sans-serif",
                fontWeight: 800,
                fontSize: "clamp(3.2rem, 8vw, 7rem)",
                lineHeight: 1.08,
                letterSpacing: "-0.02em",
                maxWidth: "820px",
              }}
            >
              <span style={{ color: "#FFFFFF" }}>Lex</span>
              <span style={{ color: "#FFD700" }}>Ved</span>
            </h1>

            {/* Subtitle */}
            <p
              style={{
                fontFamily: "'Poppins', sans-serif",
                fontWeight: 400,
                fontSize: "clamp(0.95rem, 2vw, 1.2rem)",
                color: "#8A8070",
                marginTop: "1.6rem",
                letterSpacing: "0.01em",
                maxWidth: "520px",
              }}
            >
              Your AI-powered legal research companion.
            </p>

            {/* CTA button */}
            <div style={{ marginTop: "2.8rem" }}>
              <motion.button
                id="landing-cta"
                onClick={handleClick}
                whileHover={{ scale: 1.04, backgroundColor: "#FFBE00" }}
                whileTap={{ scale: 0.97 }}
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "0.55rem",
                  padding: "0.85rem 2.2rem",
                  borderRadius: "9999px",
                  background: "#FFD700",
                  color: "#111110",
                  fontFamily: "'Poppins', sans-serif",
                  fontWeight: 700,
                  fontSize: "0.82rem",
                  letterSpacing: "0.08em",
                  textTransform: "uppercase",
                  border: "none",
                  cursor: "pointer",
                }}
              >
                Start for Free
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                  <path d="M2 7h10M8 3l4 4-4 4" stroke="#111110" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </motion.button>
            </div>

          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
