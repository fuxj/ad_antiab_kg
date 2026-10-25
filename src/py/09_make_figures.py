#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Make journal-quality figures for Chapter 6 (ARIA-H co-medication + MR direction forest).
Reads Table_S2_comedication_liability.csv and core_risk_drugs.csv from paper_tables/ and
mr_summary.csv from results/tables/. Outputs 300-dpi PNGs and vector SVGs into results/figures/."""
import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager as fm


def _use_dejavusans():
    for f in fm.findSystemFonts():
        if "DejaVuSans.ttf" in f:
            plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name()
            return
    plt.rcParams["font.family"] = "DejaVu Sans"


def find_proj_root(start: Path | None = None) -> Path:
    p = (start or Path(".")).resolve()
    for _ in range(8):
        if (p / "results" / "paper_tables" / "Table_S2_comedication_liability.csv").exists():
            return p
        p = p.parent
    raise SystemExit("cannot find project root")


def fig_6_1_core_bar(proj: Path, out_dir: Path, top_n: int = 23):
    """Fig 6.1: horizontal bar chart of adjusted liability for systemic first-line high-tier drugs."""
    _use_dejavusans()
    df = pd.read_csv(proj / "results" / "paper_tables" / "core_risk_drugs.csv")
    df = df.sort_values("liability_adj", ascending=True).tail(top_n).reset_index(drop=True)

    group_color = {
        "anticoagulant":      "#C0392B",   # deep red
        "antiplatelet_p2y12": "#E67E22",   # orange
        "nsaid_nonselective": "#7F8C8D",   # grey
        "nsaid_cox2":         "#27AE60",   # green
    }
    df["color"] = df["clinical_group"].map(lambda g: group_color.get(g, "#95A5A6"))

    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    ax.barh(np.arange(len(df)), df["liability_adj"], color=df["color"], edgecolor="white", linewidth=0.6)
    ax.set_yticks(np.arange(len(df)))
    labels = []
    for _, r in df.iterrows():
        # add bleed_genes suffix on the right if present
        t = r["active_moiety"].title()
        bg = str(r.get("bleed_genes","")) if pd.notna(r.get("bleed_genes")) else ""
        if bg and bg not in ("nan", "-"):
            t += f"  ({bg})"
        labels.append(t)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Adjusted ARIA-H co-medication liability (0–1)", fontsize=10)
    ax.set_title("Figure 6.1  Systemic first-line high-liability co-medications",
                 fontsize=11, pad=12, loc="left", weight="bold")

    # tier dashed lines
    for x, lab in [(0.75, "high tier"), (0.50, "moderate")]:
        ax.axvline(x, color="#2C3E50", linestyle="--", linewidth=0.8, alpha=0.55)
        ax.text(x + 0.005, len(df) - 0.6, lab, fontsize=7.5, color="#2C3E50", alpha=0.7)

    for i, (_, r) in enumerate(df.iterrows()):
        ax.text(r["liability_adj"] + 0.008, i, f"{r['liability_adj']:.2f}",
                va="center", fontsize=7.8, color="#2C3E50")
    if "bleed_dir_conflict" in df.columns:
        for i, (_, r) in enumerate(df.iterrows()):
            if int(r.get("bleed_dir_conflict", 0)) == 1:
                ax.text(0.01, i, " †", va="center", fontsize=9, color="#C0392B", weight="bold")
        ax.text(0.01, -1.6, "† MR direction conflicts pharmacology; liability graded on clinical pharmacology.",
                fontsize=7.2, color="#7F8C8D")

    patches = [mpatches.Patch(color=c, label=g.replace("_"," ")) for g, c in group_color.items()]
    ax.legend(handles=patches, loc="lower right", frameon=False, fontsize=8.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8.5)
    plt.tight_layout()

    for ext in ("png","svg"):
        path = out_dir / f"Fig6_1_core_liability_bar.{ext}"
        dpi = 300 if ext == "png" else None
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_dir / "Fig6_1_core_liability_bar.png"


def fig_6_2_mr_forest(proj: Path, out_dir: Path):
    """Fig 6.2: direction-aware MR forest plot for bleeding-related targets."""
    _use_dejavusans()
    mr = pd.read_csv(proj / "results" / "tables" / "mr_summary.csv")
    # pick the bleeding outcomes and genes of interest
    keep_genes = ["PTGS1","VKORC1","F10","ACE","APP"]
    keep_out = ["ich","ich_fin","sah"]
    sub = mr[(mr["gene"].isin(keep_genes)) & (mr["outcome_short"].str.lower().isin(keep_out)) & (mr["pval"] < 0.05)].copy()
    sub = sub.sort_values(["gene","pval"]).reset_index(drop=True)
    if sub.empty:
        # fallback: show the genes we do have p<0.05 with any outcome, just to avoid empty plot
        sub = mr[(mr["gene"].isin(keep_genes)) & (mr["pval"] < 0.05)].copy().sort_values(["gene","pval"]).reset_index(drop=True)

    sub["or"] = sub["OR"].astype(float)
    sub["lo"] = sub["OR_low"].astype(float)
    sub["hi"] = sub["OR_high"].astype(float)
    sub["y"] = np.arange(len(sub))[::-1]
    sub["direction"] = np.where(sub["or"] < 1, "protective (↑expr → lower risk)", "risk (↑expr → higher risk)")
    sub["color"] = np.where(sub["or"] < 1, "#2980B9", "#C0392B")

    fig, ax = plt.subplots(figsize=(8.8, max(3.8, 0.48 * len(sub) + 1.2)))
    for _, r in sub.iterrows():
        ax.plot([r["lo"], r["hi"]], [r["y"], r["y"]], color=r["color"], linewidth=1.6)
        ax.plot(r["or"], r["y"], "o", color=r["color"], markersize=6, markeredgecolor="white", markeredgewidth=0.8)
    ax.axvline(1.0, color="#2C3E50", linewidth=1.0, linestyle="-")
    ax.set_xscale("log")
    ax.set_xlabel("Odds ratio per genetically higher gene expression (log scale)", fontsize=10)
    ax.set_yticks(sub["y"])
    ax.set_yticklabels([f"{g} → {o.upper()}" for g,o in zip(sub["gene"], sub["outcome_short"])], fontsize=9)
    ax.set_title("Figure 6.2  Direction-aware MR evidence at bleeding-related drug targets",
                 fontsize=11, pad=12, loc="left", weight="bold")

    # annotation on pharmacologic implication column
    impl = {"PTGS1":" inhibit ⇒ ↑bleed", "VKORC1":" inhibit ⇒ ↑bleed",
            "F10":" direction conflict †", "ACE":" inhibit ⇒ ↓bleed risk", "APP":" (flagged)"}
    for _, r in sub.iterrows():
        ax.text(ax.get_xlim()[1] * 1.02, r["y"], impl.get(r["gene"],""), va="center", fontsize=8.2, color="#2C3E50")
    ax.text(ax.get_xlim()[1] * 1.02, len(sub) + 0.3, "pharmacological implication",
            fontsize=8, color="#7F8C8D")
    ax.set_xlim(sub["lo"].min()*0.8, sub["hi"].max()*1.8)

    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8.5)
    ax.grid(axis="x", linestyle=":", alpha=0.35)

    # legend
    prot = mpatches.Patch(color="#2980B9", label="protective (OR<1, higher expr → lower bleed)")
    risk = mpatches.Patch(color="#C0392B", label="risk (OR>1, higher expr → higher bleed)")
    ax.legend(handles=[prot, risk], loc="upper left", frameon=False, fontsize=8.5)
    ax.text(0.99, 0.01, "† MR direction for F10 contradicts clinical anticoagulation pharmacology; see text.",
            ha="right", va="bottom", transform=ax.transAxes, fontsize=7.2, color="#7F8C8D")
    plt.tight_layout()

    for ext in ("png","svg"):
        path = out_dir / f"Fig6_2_mr_bleeding_forest.{ext}"
        dpi = 300 if ext == "png" else None
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_dir / "Fig6_2_mr_bleeding_forest.png"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", default=None)
    ap.add_argument("--topn", type=int, default=23)
    args = ap.parse_args(argv)
    proj = find_proj_root(Path(args.proj) if args.proj else Path("."))
    out_dir = proj / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    p1 = fig_6_1_core_bar(proj, out_dir, top_n=args.topn)
    p2 = fig_6_2_mr_forest(proj, out_dir)
    print("[done]", p1)
    print("[done]", p2)


if __name__ == "__main__":
    sys.exit(main())
