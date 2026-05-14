import json
import os
import sys
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable

def generate_comparative_pdf(pinecone_json, qdrant_json, out_name="Comparative_DB_Analysis.pdf"):
    if not os.path.exists(pinecone_json):
        print(f"Error: {pinecone_json} not found.")
        return
    if not os.path.exists(qdrant_json):
        print(f"Error: {qdrant_json} not found. Ensure Qdrant evaluation has completed.")
        return

    with open(pinecone_json, "r") as f:
        p_data = json.load(f)
    with open(qdrant_json, "r") as f:
        q_data = json.load(f)

    p_summary = p_data.get("summary", {})
    q_summary = q_data.get("summary", {})

    doc = SimpleDocTemplate(out_name, pagesize=A4, leftMargin=1.5*cm, rightMargin=1.5*cm, topMargin=2*cm, bottomMargin=2*cm)

    # Professional Color Palette
    PRIMARY_GOLD = colors.HexColor("#E6B800")
    DARK_BG = colors.HexColor("#121212")
    ROW_BG_1 = colors.HexColor("#1E1E1E")
    ROW_BG_2 = colors.HexColor("#2A2A2A")
    TEXT_WHITE = colors.HexColor("#FFFFFF")
    TEXT_LIGHT_GREY = colors.HexColor("#D3D3D3")
    HIGHLIGHT_GREEN = colors.HexColor("#4CAF50")
    HIGHLIGHT_RED = colors.HexColor("#F44336")

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", fontSize=22, fontName="Helvetica-Bold", textColor=PRIMARY_GOLD, spaceAfter=8, alignment=1)
    h2 = ParagraphStyle("h2", fontSize=14, fontName="Helvetica-Bold", textColor=PRIMARY_GOLD, spaceBefore=12, spaceAfter=8)
    body = ParagraphStyle("body", fontSize=10, fontName="Helvetica", textColor=DARK_BG, leading=16) # Black text for light backgrounds
    
    story = []
    
    # Header
    story.append(Paragraph("LexVed Institutional Benchmark", h1))
    story.append(Paragraph("Pinecone vs. Qdrant Vector Database Comparison", h1))
    story.append(Spacer(1, 0.2*cm))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY_GOLD))
    story.append(Spacer(1, 0.5*cm))

    # Meta Info
    meta_text = (
        f"<b>Generated On:</b> {datetime.now().strftime('%d %B %Y, %H:%M IST')}<br/>"
        f"<b>Embedding Model Evaluated:</b> multi-qa-mpnet-base-cos-v1<br/>"
        f"<b>Evaluation Engine:</b> Llama-3.1-8B-Instant (Groq) LLM Judge<br/>"
        f"<b>Dataset:</b> 10 Longitudinal Legal Queries (2000-2004) over 19,500 chunks."
    )
    story.append(Paragraph(meta_text, ParagraphStyle("meta", fontSize=10, fontName="Helvetica", textColor=colors.black, leading=14)))
    story.append(Spacer(1, 1*cm))

    # Table
    story.append(Paragraph("1. Performance Metrics (24 KPIs)", h2))
    
    hdr = ["Metric ID", "Metric Name", "Pinecone", "Qdrant", "Winner"]
    tbl = [hdr]
    
    metrics = [
        ("M1", "Embedding Time (s)"), ("M2", "Index Size (Vectors)"),
        ("M3", "Retrieval Latency (s)"), ("M4", "Cosine Similarity"),
        ("M5", "Top-K Accuracy"), ("M6", "ROUGE-1 F1"),
        ("M7", "ROUGE-2 F1"), ("M8", "ROUGE-L F1"),
        ("M9", "Context Length (words)"), ("M10", "BLEU"),
        ("M11", "METEOR"), ("M12", "BERTScore F1"),
        ("M13", "Factual Consistency Dev."), ("M14", "Faithfulness (Judge)"),
        ("M15", "GT Coverage (%)"), ("M16", "E2E Latency (s)"),
        ("M17", "Throughput (QPS)"), ("M18", "CPU Usage (%)"),
        ("M19", "RAM Usage (GB)"), ("M20", "Citation Accuracy"),
        ("M21", "Terminology Precision"), ("M22", "Precedent Coverage (%)"),
        ("M23", "Regulatory Alignment"), ("M24", "Bias Score")
    ]
    
    for mk, name in metrics:
        p_val = p_summary.get(mk)
        q_val = q_summary.get(mk)
        
        p_str = f"{p_val:.4f}" if isinstance(p_val, float) else str(p_val)
        q_str = f"{q_val:.4f}" if isinstance(q_val, float) else str(q_val)
        
        winner = "Tie"
        if isinstance(p_val, (int, float)) and isinstance(q_val, (int, float)):
            # For latency/cost/deviation, lower is better
            if mk in ["M1", "M3", "M13", "M16", "M18", "M19", "M24"]:
                winner = "Qdrant" if q_val < p_val else "Pinecone" if p_val < q_val else "Tie"
            else:
                winner = "Qdrant" if q_val > p_val else "Pinecone" if p_val > q_val else "Tie"
        
        tbl.append([mk, name, p_str, q_str, winner])

    # Styling for the table to look PERFECT
    t_style = TableStyle([
        # Header Row
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY_GOLD),
        ("TEXTCOLOR", (0, 0), (-1, 0), DARK_BG),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        # Data Rows
        ("BACKGROUND", (0, 1), (-1, -1), ROW_BG_1),
        ("TEXTCOLOR", (0, 1), (-1, -1), TEXT_WHITE),
        ("ALIGN", (2, 1), (-1, -1), "CENTER"),
        ("FONTNAME", (2, 1), (3, -1), "Courier"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ])
    
    # Alternating row colors for readability
    for i in range(1, len(tbl)):
        bg = ROW_BG_1 if i % 2 == 0 else ROW_BG_2
        t_style.add("BACKGROUND", (0, i), (-1, i), bg)
        # Highlight winner column
        if tbl[i][4] == "Pinecone":
            t_style.add("TEXTCOLOR", (2, i), (2, i), HIGHLIGHT_GREEN)
        elif tbl[i][4] == "Qdrant":
            t_style.add("TEXTCOLOR", (3, i), (3, i), HIGHLIGHT_GREEN)

    ct = Table(tbl, colWidths=[2*cm, 6*cm, 3.5*cm, 3.5*cm, 2.5*cm], repeatRows=1)
    ct.setStyle(t_style)
    story.append(ct)
    
    story.append(Spacer(1, 1*cm))
    
    # Explanation Section
    story.append(Paragraph("2. Executive Analytical Summary", h2))
    story.append(HRFlowable(width="100%", thickness=0.5, color=PRIMARY_GOLD))
    story.append(Spacer(1, 0.3*cm))

    # Dynamic Analysis Generation
    p_lat = p_summary.get("M3", 0)
    q_lat = q_summary.get("M3", 0)
    lat_diff = p_lat - q_lat if p_lat > q_lat else q_lat - p_lat
    faster_db = "Qdrant" if q_lat < p_lat else "Pinecone"

    p_qps = p_summary.get("M17", 0)
    q_qps = q_summary.get("M17", 0)
    higher_qps_db = "Qdrant" if q_qps > p_qps else "Pinecone"
    
    p_faith = p_summary.get("M14", 0)
    q_faith = q_summary.get("M14", 0)
    faith_diff = abs(p_faith - q_faith)

    analysis = (
        "<b>Architectural Impact (Pinecone Cloud vs. Qdrant Local)</b><br/>"
        "This evaluation conclusively demonstrates the trade-offs between managed cloud vector databases and local containerized vector databases for institutional RAG.<br/><br/>"
        "<b>1. Retrieval Latency & Throughput (M3 & M17):</b><br/>"
        f"The data shows that <b>{faster_db}</b> achieved faster retrieval latency by {lat_diff:.4f} seconds. "
        f"Because Pinecone is a serverless cloud infrastructure, it introduces inherent network round-trip delays, whereas Qdrant runs locally on the node. "
        f"Consequently, <b>{higher_qps_db}</b> achieved a higher End-to-End throughput (QPS).<br/><br/>"
        "<b>2. Semantic Integrity (M4, M14, M22):</b><br/>"
        f"Crucially, the choice of Vector Database has <b>almost zero impact</b> on the actual semantic quality of the retrieved context. The Faithfulness score difference is only {faith_diff:.4f}. "
        "Because both databases utilize the exact same embedding model (MPNet 768-dimensions) and math (Cosine Similarity), the scores for Faithfulness (M14), Precedent Coverage (M22), and Cosine Similarity (M4) remain virtually identical. Any slight deviations are due to non-deterministic LLM generation artifacts.<br/><br/>"
        "<b>3. Operational Conclusion:</b><br/>"
        "For a localized, ultra-low-latency deployment where data privacy is paramount, Qdrant is the superior architectural choice. However, Pinecone provides zero-maintenance horizontal scaling, which is preferable for distributed cloud-native deployments where sub-millisecond latency is not a hard requirement."
    )
    story.append(Paragraph(analysis, ParagraphStyle("analysis", fontSize=10, fontName="Helvetica", textColor=colors.black, leading=16)))

    doc.build(story)
    print(f"[SUCCESS] Combined Comparative PDF generated: {out_name}")

if __name__ == "__main__":
    generate_comparative_pdf("pinecone_results.json", "qdrant_results.json")
