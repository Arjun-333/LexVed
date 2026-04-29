"use client";

import { motion } from "framer-motion";

interface WelcomeHeroProps {
  onSuggestionClick: (text: string) => void;
}

const features = [
  {
    icon: "history_edu",
    title: "Criminal Jurisprudence",
    desc: "Search across section 302/307 IPC, bail precedents, and criminal appeals with precision.",
    query: "Explain Section 302 IPC with landmark cases",
    color: "var(--accent)",
  },
  {
    icon: "gavel",
    title: "Civil Litigation",
    desc: "Analyze property disputes, contract breaches, and civil procedure from indexed documents.",
    query: "Abhishek Banerjee Case summary",
    color: "var(--accent)",
  },
  {
    icon: "verified_user",
    title: "Statutory Compliance",
    desc: "Cross-reference statutes and automated domain detection for high-fidelity citations.",
    query: "Bail Grounds 2024",
    color: "var(--accent)",
  },
];

export default function WelcomeHero({ onSuggestionClick }: WelcomeHeroProps) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center px-6">
      <motion.div
        initial={{ opacity: 0, scale: 0.98 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
        className="text-center mb-16"
      >
        <div
          className="w-20 h-20 rounded-2xl flex items-center justify-center mx-auto mb-8 transition-all duration-700"
          style={{
            background: "var(--accent-bg)",
            boxShadow: "var(--shadow-gold), inset 0 0 20px rgba(255,255,255,0.02)",
            border: "1px solid var(--border)",
          }}
        >
          <span className="material-icons-round text-[38px]" style={{ color: "var(--accent)" }}>
            balance
          </span>
        </div>
        <h1
          className="text-5xl md:text-6xl font-bold tracking-tight mb-4 transition-colors duration-300"
          style={{ fontFamily: "var(--font-serif)", color: "var(--text)" }}
        >
          Lex<span style={{ color: "var(--accent)" }}>Ved</span>
        </h1>
        <p className="text-[1.1rem] transition-colors duration-300 max-w-[550px] mx-auto leading-relaxed italic"
           style={{ fontFamily: "var(--font-serif)", color: "var(--text-secondary)" }}>
          "The most sophisticated local intelligence layer for the modern litigator."
        </p>
      </motion.div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-[900px] w-full px-4">
        {features.map((f, i) => (
          <motion.button
            key={f.title}
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.3 + i * 0.1, ease: [0.16, 1, 0.3, 1] }}
            onClick={() => onSuggestionClick(f.query)}
            className="group text-left p-7 rounded-[24px] cursor-pointer transition-all duration-400 relative overflow-hidden"
            style={{
              background: "var(--surface)",
              border: "1px solid var(--border)",
              boxShadow: "var(--shadow-prestige)",
            }}
          >
            {/* Soft Ambient Glow */}
            <div className="absolute top-0 right-0 w-24 h-24 blur-3xl opacity-0 group-hover:opacity-10 transition-opacity duration-700"
                 style={{ background: f.color }} />

            <div
              className="w-11 h-11 rounded-xl flex items-center justify-center mb-5 transition-all duration-400
                       group-hover:scale-110"
              style={{ background: `${f.color}10`, color: f.color, border: "1px solid rgba(0,0,0,0.03)" }}
            >
              <span className="material-icons-round text-[22px]">{f.icon}</span>
            </div>

            <h3
              className="font-bold text-[0.95rem] mb-2 transition-colors duration-300 group-hover:text-[var(--accent)]"
              style={{ color: "var(--text)" }}
            >
              {f.title}
            </h3>
            <p
              className="text-[0.78rem] leading-[1.65] transition-colors duration-300"
              style={{ color: "var(--text-muted)" }}
            >
              {f.desc}
            </p>

            <div className="mt-5 flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-all duration-500 translate-x-[-10px] group-hover:translate-x-0">
               <span className="text-[0.6rem] font-bold uppercase tracking-[0.15em]" style={{ color: "var(--accent)" }}>
                 Initiate Brief
               </span>
               <span className="material-icons-round text-[14px]" style={{ color: "var(--accent)" }}>
                 north_east
               </span>
            </div>
          </motion.button>
        ))}
      </div>
    </div>
  );
}
