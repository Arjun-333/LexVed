"""
LexVed Primitive Pipeline — All Models Runner + PDF Report
Runs MPNet, MiniLM, DistilBERT sequentially via Pinecone + Groq.
BGE-M3 skipped unless FlagEmbedding is installed.
"""
import os, sys, time, json
from pathlib import Path
from datetime import datetime

BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(BACKEND_DIR)

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle, Paragraph,
                                 Spacer, HRFlowable, PageBreak)
from reportlab.lib.enums import TA_CENTER

from primitive_pipeline import run_primitive_pipeline

MODELS = [
    ("1", "MPNet"),
]

# Try BGE-M3 only if FlagEmbedding available
pass

METRIC_LABELS = [
    ("M1",  "Embedding Time",           "s",  4),
    ("M2",  "Index Size (Vectors)",      "",   0),
    ("M3",  "Retrieval Latency",         "s",  4),
    ("M4",  "Cosine Similarity",         "",   4),
    ("M5",  "Top-K Accuracy",            "",   2),
    ("M6",  "ROUGE-1 F1",               "",   4),
    ("M7",  "ROUGE-2 F1",               "",   4),
    ("M8",  "ROUGE-L F1",               "",   4),
    ("M9",  "Context Length",            "w",  0),
    ("M10", "BLEU",                      "",   4),
    ("M11", "METEOR",                    "",   4),
    ("M12", "BERTScore F1",              "",   4),
    ("M13", "Factual Consistency Dev.",   "",   4),
    ("M14", "Faithfulness (Judge)",       "",   4),
    ("M15", "GT Coverage (Judge)",        "%",  2),
    ("M16", "E2E Latency",              "s",  4),
    ("M17", "Throughput",                "QPS",4),
    ("M18", "CPU Usage",                 "%",  1),
    ("M19", "RAM Usage",                 "GB", 2),
    ("M20", "Citation Accuracy (Judge)",  "",   4),
    ("M21", "Term. Precision (Judge)",    "",   4),
    ("M22", "Precedent Coverage (Judge)", "%",  2),
    ("M23", "Reg. Alignment (Judge)",     "",   4),
    ("M24", "Bias Score (Judge)",         "",   4),
]

print("\n" + "="*70)
print("  LexVed Primitive Pipeline — All Models Audit")
print("  Pinecone + Groq llama-3.1-8b-instant")
print("="*70)

results = {}
sys_info = {}

for choice, label in MODELS:
    print(f"\n{'─'*60}\n  Running: {label}\n{'─'*60}")
    t0 = time.time()
    df = run_primitive_pipeline(model_choice=choice)
    elapsed = time.time() - t0
    if df is not None:
        results[label] = df
        # Read system_info from status file
        try:
            rpt = json.loads((BACKEND_DIR / "primitive_evaluation_results.json").read_text())
            si = rpt.get("system_info", {})
        except: si = {}
        sys_info[label] = {**si, "elapsed_s": round(elapsed, 1)}
        print(f"  ✓ {label} done in {elapsed:.0f}s")
    else:
        print(f"  ✗ {label} FAILED")

if not results:
    print("ERROR: No models produced results."); sys.exit(1)

# ── Build summary ─────────────────────────────────────────────────────
summary = {}
for label, df in results.items():
    rpt = json.loads((BACKEND_DIR / "primitive_evaluation_results.json").read_text())
    summary[label] = rpt.get("summary", {})

# ── Generate PDF ──────────────────────────────────────────────────────
out_pdf = BACKEND_DIR / f"primitive_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
doc = SimpleDocTemplate(str(out_pdf), pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm, topMargin=2.5*cm, bottomMargin=2*cm)

GOLD = colors.HexColor("#D4AF37"); DARK = colors.HexColor("#0A0A0A")
GREY = colors.HexColor("#1A1A1A"); DIM = colors.HexColor("#666666")

h1 = ParagraphStyle("h1", fontSize=18, fontName="Helvetica-Bold",
                    textColor=GOLD, spaceAfter=4, alignment=TA_CENTER)
h2 = ParagraphStyle("h2", fontSize=12, fontName="Helvetica-Bold",
                    textColor=GOLD, spaceBefore=12, spaceAfter=4)
body = ParagraphStyle("body", fontSize=8, fontName="Helvetica",
                      textColor=colors.HexColor("#CCCCCC"), spaceAfter=4)
mono = ParagraphStyle("mono", fontSize=7, fontName="Courier",
                      textColor=colors.HexColor("#BBBBBB"))
csm = ParagraphStyle("csm", fontSize=7, fontName="Helvetica",
                      textColor=DIM, alignment=TA_CENTER)

model_labels = list(results.keys())
story = []

# Cover
story.append(Spacer(1, 2*cm))
story.append(Paragraph("LexVed", h1))
story.append(Paragraph("PRIMITIVE PIPELINE — INSTITUTIONAL AUDIT", h1))
story.append(Spacer(1, 0.3*cm))
story.append(HRFlowable(width="100%", thickness=1, color=GOLD))
story.append(Spacer(1, 0.4*cm))

first_si = list(sys_info.values())[0]
meta = [
    ["Generated",      datetime.now().strftime("%d %B %Y, %H:%M IST")],
    ["Models",         ", ".join(model_labels)],
    ["Queries",        "10 (5 Civil + 5 Criminal)"],
    ["PDF Corpus",     f"{first_si.get('total_pdfs','?')} PDFs"],
    ["Chunks",         f"{first_si.get('total_chunks','?')}"],
    ["Vector DB",      f"Pinecone (per-model indexes)"],
    ["Generator",      "Groq llama-3.1-8b-instant"],
    ["Judge (M14-M24)","Groq llama-3.1-8b-instant"],
]
mt = Table(meta, colWidths=[4.5*cm, 12*cm])
mt.setStyle(TableStyle([
    ("FONTNAME",(0,0),(-1,-1),"Helvetica"), ("FONTSIZE",(0,0),(-1,-1),8),
    ("TEXTCOLOR",(0,0),(0,-1),GOLD), ("TEXTCOLOR",(1,0),(1,-1),colors.HexColor("#CCC")),
    ("BACKGROUND",(0,0),(-1,-1),GREY), ("ROWBACKGROUNDS",(0,0),(-1,-1),[GREY,DARK]),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#333")),
    ("LEFTPADDING",(0,0),(-1,-1),8), ("TOPPADDING",(0,0),(-1,-1),4),
    ("BOTTOMPADDING",(0,0),(-1,-1),4),
]))
story.append(mt)
story.append(PageBreak())

# Comparative table
story.append(Paragraph("1. Comparative Summary — 24 KPIs", h2))
story.append(HRFlowable(width="100%", thickness=0.5, color=GOLD))
story.append(Spacer(1, 0.3*cm))

col_w = [0.9*cm, 4.2*cm] + [3.0*cm]*len(model_labels)
hdr = ["ID", "Metric"] + model_labels
tbl = [hdr]
for mid, lbl, unit, dec in METRIC_LABELS:
    row = [mid, f"{lbl}" + (f" ({unit})" if unit else "")]
    for ml in model_labels:
        v = summary.get(ml, {}).get(mid)
        row.append(f"{v:.{dec}f}" if v is not None else "—")
    tbl.append(row)

ct = Table(tbl, colWidths=col_w, repeatRows=1)
ct.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(-1,0),GOLD), ("TEXTCOLOR",(0,0),(-1,0),DARK),
    ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"), ("FONTSIZE",(0,0),(-1,-1),7),
    ("FONTNAME",(2,1),(-1,-1),"Courier"),
    ("TEXTCOLOR",(0,1),(0,-1),GOLD), ("TEXTCOLOR",(1,1),(1,-1),colors.HexColor("#BBB")),
    ("TEXTCOLOR",(2,1),(-1,-1),colors.HexColor("#DDD")),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[DARK,GREY]),
    ("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#333")),
    ("ALIGN",(2,0),(-1,-1),"CENTER"), ("ALIGN",(0,0),(0,-1),"CENTER"),
    ("TOPPADDING",(0,0),(-1,-1),3), ("BOTTOMPADDING",(0,0),(-1,-1),3),
]))
story.append(ct)
story.append(Spacer(1, 0.5*cm))

story.append(Paragraph(
    f"Report by LexVed · {datetime.now().strftime('%Y-%m-%d %H:%M')} IST · "
    "Primitive Pipeline (Pinecone) · Groq LLM Judge",
    csm
))

doc.build(story)
print(f"\n{'='*70}")
print(f"  PDF REPORT → {out_pdf}")
print(f"{'='*70}\n")
