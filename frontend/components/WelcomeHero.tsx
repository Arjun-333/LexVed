"use client";

import { motion } from "framer-motion";

interface WelcomeHeroProps {
  onSuggestionClick: (text: string) => void;
}

const suggestions = [
  {
    icon: "gavel",
    title: "Criminal Jurisprudence",
    desc: "Search across IPC sections, bail precedents, and criminal appeals with precision.",
    query: "Explain Section 302 IPC with landmark cases",
  },
  {
    icon: "balance",
    title: "Civil Litigation",
    desc: "Analyze property disputes, contract breaches, and civil procedure from indexed documents.",
    query: "Abhishek Banerjee Case summary",
  },
  {
    icon: "verified_user",
    title: "Statutory Compliance",
    desc: "Cross-reference statutes and automated domain detection for high-fidelity citations.",
    query: "Bail Grounds 2024",
  },
];

export default function WelcomeHero({ onSuggestionClick }: WelcomeHeroProps) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center px-8 py-6 relative">

      {/* Suggestion Cards — bottom area */}
      <motion.div
        initial={{ opacity: 0, y: 18 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.15, ease: [0.16, 1, 0.3, 1] }}
        className="flex flex-col md:flex-row gap-3 max-w-[760px] w-full"
      >
        {suggestions.map((s, i) => (
          <motion.button
            key={s.title}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45, delay: 0.2 + i * 0.08, ease: [0.16, 1, 0.3, 1] }}
            onClick={() => onSuggestionClick(s.query)}
            className="group flex-1 text-left p-5 rounded-xl cursor-pointer transition-all duration-250 relative overflow-hidden"
            style={{
              background: "#111111",
              border: "1px solid #1e1e1e",
            }}
          >
            {/* Hover gold border flash */}
            <div className="absolute inset-0 rounded-xl border border-[#D4AF37] opacity-0 group-hover:opacity-20 transition-opacity duration-250 pointer-events-none" />

            <div
              className="w-8 h-8 rounded-lg flex items-center justify-center mb-3 transition-transform duration-300 group-hover:scale-105"
              style={{ background: "rgba(212,175,55,0.08)", color: "#D4AF37" }}
            >
              <span className="material-icons-round text-[17px]">{s.icon}</span>
            </div>

            <h3
              className="font-semibold text-[0.85rem] mb-1.5 transition-colors duration-200 group-hover:text-[#D4AF37]"
              style={{ color: "#d0d0d0", letterSpacing: "-0.02em" }}
            >
              {s.title}
            </h3>
            <p className="text-[0.73rem] leading-[1.6]" style={{ color: "#555555" }}>
              {s.desc}
            </p>

            <div className="mt-3 flex items-center gap-1.5 opacity-0 group-hover:opacity-100 transition-opacity duration-250">
              <span className="text-[0.58rem] font-bold uppercase tracking-[0.1em]" style={{ color: "#D4AF37" }}>
                Ask this
              </span>
              <span className="material-icons-round text-[12px]" style={{ color: "#D4AF37" }}>
                arrow_forward
              </span>
            </div>
          </motion.button>
        ))}
      </motion.div>
    </div>
  );
}
