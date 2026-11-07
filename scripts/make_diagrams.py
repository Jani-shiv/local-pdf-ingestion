"""Generate the original diagrams used in the article (docs/images/*.png)."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent.parent / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)
INK, BG = "#0f172a", "#ffffff"
COLORS = ["#2563eb", "#7c3aed", "#0d9488", "#ea580c", "#475569"]


def canvas(w=12, h=6.2):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    fig.patch.set_facecolor(BG)
    return fig, ax


def box(ax, x, y, w, h, text, color, sub=None, fs=14):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=2.5",
                                fc=color, ec="none"))
    ax.text(x + w / 2, y + h / 2 + (3 if sub else 0), text, ha="center", va="center",
            color="white", fontsize=fs, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 5, sub, ha="center", va="center", color="white", fontsize=10.5)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18,
                                 color=INK, lw=2))


def pipeline():
    fig, ax = canvas()
    ax.text(50, 92, "PDF ingestion pipeline", ha="center", fontsize=22, fontweight="bold", color=INK)
    ax.text(50, 84, "Everything runs on your own machine", ha="center", fontsize=13, color="#475569")
    steps = [("PDF", "documents/*.pdf"), ("Extract", "PyMuPDF"), ("Chunk", "1000 chars\n150 overlap"),
             ("Embed", "Ollama\nnomic-embed-text"), ("Store", "JSON file")]
    w, gap, x = 14.5, 5.5, 2.5
    for i, (t, s) in enumerate(steps):
        box(ax, x, 38, w, 28, t, COLORS[i], s)
        if i < len(steps) - 1:
            arrow(ax, x + w + 0.8, 52, x + w + gap - 0.8, 52)
        x += w + gap
    ax.text(50, 22, "This article builds the knowledge layer. Search and chat come in the next article.",
            ha="center", fontsize=12, color="#475569")
    fig.savefig(OUT / "01-pipeline.png", bbox_inches="tight"); plt.close(fig)


def chunking():
    fig, ax = canvas(12, 5.6)
    ax.text(50, 94, "How overlapping chunks work", ha="center", fontsize=22, fontweight="bold", color=INK)
    ax.add_patch(Rectangle((5, 66), 90, 10, fc="#e2e8f0", ec="none"))
    ax.text(20, 71, "Page text (one long string)", ha="center", va="center", fontsize=12, color=INK)
    spans = [(5, 41, "Chunk 0", COLORS[0], 44), (35, 71, "Chunk 1", COLORS[1], 28), (65, 95, "Chunk 2", COLORS[2], 12)]
    for x1, x2, label, c, y in spans:
        ax.add_patch(Rectangle((x1, y), x2 - x1, 10, fc=c, ec="none", alpha=0.92))
        ax.text((x1 + x2) / 2, y + 5, label, ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    for x1, x2 in [(35, 41), (65, 71)]:
        ax.add_patch(Rectangle((x1, 10), x2 - x1, 62, fc="none", ec="#dc2626", lw=1.8, ls="--"))
    ax.text(38, 5, "overlap", ha="center", color="#dc2626", fontsize=12, fontweight="bold")
    ax.text(68, 5, "overlap", ha="center", color="#dc2626", fontsize=12, fontweight="bold")
    fig.savefig(OUT / "02-chunking-overlap.png", bbox_inches="tight"); plt.close(fig)


def embeddings():
    fig, ax = canvas(12, 6)
    ax.text(50, 94, "Different words, similar meaning", ha="center", fontsize=22, fontweight="bold", color=INK)
    texts = ["How do I restart a Kubernetes pod?", "Steps to recreate a failed pod", "Best pasta recipe for dinner"]
    cols = [COLORS[0], COLORS[0], "#94a3b8"]
    ys = [66, 44, 22]
    vec = [[0.82, 0.11, 0.64], [0.79, 0.14, 0.61], [0.05, 0.93, 0.12]]
    for t, c, y, v in zip(texts, cols, ys, vec):
        box(ax, 3, y, 44, 14, t, c, fs=12)
        arrow(ax, 48, y + 7, 56, y + 7)
        ax.text(57, y + 7, "[" + ", ".join(f"{n:.2f}" for n in v) + ", ...]", va="center",
                fontsize=13, family="monospace", color=INK)
    ax.text(50, 6, "Illustrative numbers only. Real nomic-embed-text vectors are much longer.",
            ha="center", fontsize=11, color="#475569")
    fig.savefig(OUT / "03-embeddings-concept.png", bbox_inches="tight"); plt.close(fig)


def cover():
    fig, ax = canvas(12, 6.3)
    fig.patch.set_facecolor("#0f172a")
    ax.add_patch(Rectangle((0, 0), 100, 100, fc="#0f172a", ec="none"))
    ax.text(6, 78, "Turn PDFs into", fontsize=36, fontweight="bold", color="white")
    ax.text(6, 63, "searchable knowledge", fontsize=36, fontweight="bold", color="#60a5fa")
    ax.text(6, 48, "for your local AI", fontsize=36, fontweight="bold", color="white")
    ax.text(6, 30, "Python + PyMuPDF + Ollama  |  Step 1 of a local RAG series", fontsize=15, color="#cbd5e1")
    ax.text(6, 14, "Shiv Jani", fontsize=18, fontweight="bold", color="#facc15")
    for i, c in enumerate(COLORS):
        ax.add_patch(Rectangle((68 + i * 5.5, 12), 4.5, 4.5, fc=c, ec="none"))
    fig.savefig(OUT / "00-cover.png", bbox_inches="tight", facecolor=fig.get_facecolor()); plt.close(fig)


for fn in (cover, pipeline, chunking, embeddings):
    fn()
print(sorted(p.name for p in OUT.glob("*.png")))
