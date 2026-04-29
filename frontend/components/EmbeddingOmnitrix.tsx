"use client";

import { useEffect, useRef } from "react";
import { motion } from "framer-motion";

const embeddingModels = [
  { id: "multi-qa-mpnet-base-cos-v1", name: "MPNet", desc: "768 DIM • DENSE", icon: "hub" },
  { id: "multi-qa-MiniLM-L6-cos-v1", name: "MiniLM", desc: "384 DIM • LITE", icon: "architecture" },
  { id: "multi-qa-distilbert-cos-v1", name: "DistilBERT", desc: "768 DIM • BALANCED", icon: "psychology" },
  { id: "BAAI/bge-m3", name: "BGE-M3", desc: "1024 DIM • MULTILINGUAL", icon: "language" },
  { id: "intfloat/multilingual-e5-large-instruct", name: "E5-Mistral", desc: "4096 DIM • INSTRUCT", icon: "auto_awesome" },
  { id: "Cohere/Cohere-embed-english-v3.0", name: "Cohere", desc: "1024 DIM • API", icon: "cloud" },
];

interface Props {
  selectedModel: string;
  onSelect: (id: string) => void;
}

export default function EmbeddingOmnitrix({ selectedModel, onSelect }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const observerRef = useRef<IntersectionObserver | null>(null);

  // Use IntersectionObserver to seamlessly update the selected state without blocking the animation thread
  useEffect(() => {
    const container = scrollRef.current;
    if (!container) return;

    observerRef.current = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const modelId = entry.target.getAttribute("data-model-id");
            if (modelId && modelId !== selectedModel) {
              onSelect(modelId);
            }
          }
        });
      },
      { root: container, threshold: 0.6 } // Needs to be 60% visible to become 'selected'
    );

    const cards = container.querySelectorAll(".model-card");
    cards.forEach((card) => observerRef.current?.observe(card));

    return () => observerRef.current?.disconnect();
  }, [selectedModel, onSelect]);

  // Initial scroll to the selected item on mount
  useEffect(() => {
    const container = scrollRef.current;
    if (!container) return;
    const activeCard = container.querySelector(`[data-model-id="${selectedModel}"]`) as HTMLElement;
    if (activeCard) {
      container.scrollTo({
        left: activeCard.offsetLeft - container.offsetWidth / 2 + activeCard.offsetWidth / 2,
        behavior: "smooth"
      });
    }
  }, []);

  return (
    <div className="w-full pt-6 pb-2">
      <div className="flex flex-col items-center gap-2 mb-6">
        <h3 className="font-display text-lg tracking-[0.3em] text-[#D4AF37] uppercase">
          Neural Architecture
        </h3>
        <div className="h-px w-12 bg-gradient-to-r from-transparent via-[#D4AF37] to-transparent opacity-50" />
      </div>

      {/* Hardware-Accelerated Native Scroll Container */}
      <div 
        ref={scrollRef}
        className="flex overflow-x-auto gap-6 py-8 px-[calc(50%-100px)] snap-x snap-mandatory scroll-smooth [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none] cursor-grab active:cursor-grabbing"
        style={{ scrollBehavior: 'smooth' }}
      >
        {embeddingModels.map((m) => {
          const isSelected = m.id === selectedModel;
          return (
            <button
              key={m.id}
              data-model-id={m.id}
              onClick={() => {
                const container = scrollRef.current;
                const card = container?.querySelector(`[data-model-id="${m.id}"]`) as HTMLElement;
                if (container && card) {
                  container.scrollTo({
                    left: card.offsetLeft - container.offsetWidth / 2 + card.offsetWidth / 2,
                    behavior: "smooth"
                  });
                }
              }}
              className={`model-card relative flex-shrink-0 w-[200px] h-[160px] flex flex-col items-center justify-center border transition-all duration-500 ease-out snap-center
                ${isSelected 
                  ? 'border-[#D4AF37] bg-black/80 scale-110 shadow-[0_0_30px_rgba(212,175,55,0.2)] z-10' 
                  : 'border-[#D4AF37]/20 bg-transparent scale-90 opacity-40 hover:opacity-70 hover:border-[#D4AF37]/50'
                }
              `}
            >
              {/* Corner Ornaments */}
              {isSelected && (
                <>
                  <div className="absolute top-2 left-2 w-2 h-2 border-t border-l border-[#D4AF37] opacity-60" />
                  <div className="absolute bottom-2 right-2 w-2 h-2 border-b border-r border-[#D4AF37] opacity-60" />
                </>
              )}

              <span className={`material-icons-round transition-all duration-500 ${isSelected ? 'text-4xl text-[#D4AF37] mb-3' : 'text-2xl text-white/40 mb-2'}`}>
                {m.icon}
              </span>
              
              <h4 className={`font-serif tracking-widest uppercase transition-colors duration-500 ${isSelected ? 'text-lg text-[#D4AF37]' : 'text-xs text-white/60'}`}>
                {m.name}
              </h4>

              {isSelected && (
                <div className="mt-3 flex flex-col items-center">
                  <div className="h-px w-8 bg-[#D4AF37]/40 mb-1" />
                  <p className="text-[8px] font-mono tracking-[0.2em] text-[#D4AF37] uppercase">
                    {m.desc}
                  </p>
                </div>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
