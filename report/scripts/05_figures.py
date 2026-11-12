"""Figure and the numbers quoted in the pysccdc technical report.

Usage: python 05_figures.py <runs_dir> <tasks_dir> <out_dir>
Compares scCDC 1.4 and pysccdc on the 10x Genomics human-mouse mixtures. In a mixture, counts of mouse genes
in human cells (and of human genes in mouse cells) can only come from ambient RNA, which shows how much of the
contamination the correction removes and how much of the cells' own RNA it takes with it.
"""
from __future__ import annotations

import gzip
import json
import pathlib
import sys

import anndata as ad
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy
import scipy.io
import scipy.sparse as sp

import pysccdc

RUNS, TASKS, OUT = map(pathlib.Path, sys.argv[1:4])
OUT.mkdir(parents=True, exist_ok=True)
ASSETS = pathlib.Path(__file__).resolve().parent.parent / "assets"
C_R, C_PY = "#eb6834", "#2a78d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2, "ytick.major.size": 2, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 300,
})
ENG = {"R": "scCDC", "py": "pysccdc"}
LABEL = {"hgmm_1k_v3": "hgmm 1k (v3)", "hgmm_5k_v3": "hgmm 5k (v3)", "hgmm_6k": "hgmm 6k (v2)",
         "hgmm_10k_v3": "hgmm 10k (v3)", "hgmm_12k": "hgmm 12k (v2)"}
EXAMPLE = "hgmm_12k"


def result(engine, task, run=1):
    d = RUNS / ENG[engine] / task / f"run{run}"
    if not (d / "result.json").exists():
        return None
    r = json.load(open(d / "result.json"))
    g = pd.read_csv(d / "gcg.tsv", sep="\t")
    m = sp.csc_matrix((np.fromfile(d / "corr_x.bin"), np.fromfile(d / "corr_i.bin", np.int32),
                       np.fromfile(d / "corr_p.bin", np.int32)), shape=(len(g), r["dim"][1]))
    return r, g, m


rows, cellwise, umap_frac, dist_pairs, removed_pairs = [], {}, {}, [], []
failed = []
for t in LABEL:
    td = TASKS / t
    genes = pd.read_csv(td / "genes.tsv", sep="\t")
    species = pd.read_csv(td / "species.tsv", sep="\t")
    droplets = (td / "droplets.tsv").read_text().split()
    tod = sp.csc_matrix(scipy.io.mmread(gzip.open(td / "tod.mtx.gz")))
    cl = pd.read_csv(td / "clusters.tsv", sep="\t", dtype=str)
    toc = tod[:, pd.Index(droplets).get_indexer(cl.cell)].tocsr().astype(np.float64)
    res = {e: result(e, t) for e in ENG}
    if res["R"] is None or res["py"] is None:
        failed.append(t)
        continue
    sp_of = species.set_index("cell").species.reindex(cl.cell).to_numpy()
    mouse_gene = (genes.genome == "mouse").to_numpy()
    hum, mou = sp_of == "human", sp_of == "mouse"
    pure = hum | mou

    def split(m):
        m_cnt = np.asarray(m[mouse_gene].sum(axis=0)).ravel()
        h_cnt = np.asarray(m[~mouse_gene].sum(axis=0)).ravel()
        return (np.where(hum, m_cnt, np.where(mou, h_cnt, np.nan)),
                np.where(hum, h_cnt, np.where(mou, m_cnt, np.nan)))

    o0, s0 = split(toc)
    row = dict(dataset=t, label=LABEL[t], droplets=tod.shape[1], cells=toc.shape[1],
               other_share=o0[pure].sum() / (o0 + s0)[pure].sum())
    for e in ENG:
        r, g, m = res[e]
        gi = g.row.to_numpy() - 1
        delta = sp.csr_matrix(toc[gi]) - sp.csr_matrix(m)          # counts removed, GCG rows x cells
        rem = sp.csr_matrix((delta.data, delta.indices, delta.indptr), shape=delta.shape)
        dm = np.asarray(rem[mouse_gene[gi]].sum(axis=0)).ravel()
        dh = np.asarray(rem[~mouse_gene[gi]].sum(axis=0)).ravel()
        o_rem = np.where(hum, dm, np.where(mou, dh, np.nan))
        s_rem = np.where(hum, dh, np.where(mou, dm, np.nan))
        row[f"n_gcg_{e}"] = len(g)
        row[f"removed_{e}"] = o_rem[pure].sum() / o0[pure].sum()
        row[f"lost_{e}"] = s_rem[pure].sum() / s0[pure].sum()
        row[f"gcg_mouse_{e}"] = int(mouse_gene[gi].sum())
        row[f"sec_{e}"] = min(json.load(open(RUNS / ENG[e] / t / f"run{k}" / "result.json"))["seconds"]
                              for k in (1, 2))
        if t == EXAMPLE:
            o1, s1 = o0 - o_rem, s0 - s_rem
            cellwise[e] = (o0[pure] / (o0 + s0)[pure], o1[pure] / (o1 + s1)[pure])
            umap_frac[e] = np.where(pure, o1 / (o1 + s1), np.nan)
    gR, gP = res["R"][1], res["py"][1]
    row["same_gcg"] = list(gR.row) == list(gP.row)
    row["dist_maxdiff"] = float(np.abs(gR.iloc[:, 1:].to_numpy() - gP.iloc[:, 1:].to_numpy()).max())
    row["corr_maxdiff"] = float(abs(res["R"][2] - res["py"][2]).max())
    row["others_unchanged"] = bool(res["R"][0]["others_unchanged"] and res["py"][0]["others_unchanged"])
    dist_pairs.append(np.column_stack([gR.mean_distance, gP.mean_distance]))
    gi = gR.row.to_numpy() - 1
    removed_pairs.append(np.column_stack([np.asarray((toc[gi] - res["R"][2]).sum(axis=1)).ravel(),
                                          np.asarray((toc[gi] - res["py"][2]).sum(axis=1)).ravel()]))
    if t == EXAMPLE:
        umap_frac["before"] = np.where(pure, o0 / (o0 + s0), np.nan)
        umap_species = np.where(pd.isna(sp_of), "mixed", sp_of)
        a = scanpy.AnnData(sp.csr_matrix(toc.T).astype(np.float32))
        scanpy.pp.normalize_total(a, target_sum=1e4)
        scanpy.pp.log1p(a)
        scanpy.pp.highly_variable_genes(a, n_top_genes=2000, subset=True)
        scanpy.pp.pca(a, n_comps=30, random_state=0)
        scanpy.pp.neighbors(a, n_neighbors=15, random_state=0)
        scanpy.tl.umap(a, random_state=0)
        xy = a.obsm["X_umap"]
        # entropy-expression curve of the largest cluster (pysccdc diagnostics, identical to R)
        keep = np.flatnonzero(np.diff((toc != 0).tocsr().indptr) >= 3)
        adata = ad.AnnData(sp.csr_matrix(toc[keep].T))
        adata.var_names = [f"g{i + 1}" for i in keep]
        adata.obs["cluster"] = pd.Categorical(cl.cluster.to_numpy(), categories=sorted(cl.cluster.unique(), key=int))
        full = pysccdc.ContaminationDetection(adata, cluster_key="cluster", return_full=True)
        big = cl.cluster.value_counts().index[0]
        curve = full["per_cluster"][big]
        curve_gcg = set(full["GCGs"])
        curve_name = big
    rows.append(row)
tab = pd.DataFrame(rows)
tab.to_csv(OUT / "comparison_per_dataset.tsv", sep="\t", index=False)
dist_pairs, removed_pairs = np.vstack(dist_pairs), np.vstack(removed_pairs)
tab["remc"] = tab.removed_R * tab.other_share * 100
tab["lostc"] = tab.lost_R * (1 - tab.other_share) * 100

# ======================= Figure 1 =======================
fig = plt.figure(figsize=(7.2, 6.5))
gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.95], hspace=0.55)


def panel(ax, s, dx=-0.36):
    ax.text(dx, 1.02, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left", color=INK)


def crop(img):
    ink = (img[..., 3] > 0.04) & (img[..., :3].min(-1) < 0.96)
    ys, xs = np.where(ink)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


top = gs[0].subgridspec(1, 4, width_ratios=[0.62, 1, 1, 1], wspace=0.55)
logos = [crop(plt.imread(ASSETS / f)) for f in ["pysccdc_logo.png", "omicverse_logo.png"]]
x = np.arange(len(tab))

axa = fig.add_subplot(top[1])
w = 0.26
axa.bar(x - w, tab.other_share * 100, w * 0.9, color="#c9c8c3", label="contamination present")
axa.bar(x, tab.remc, w * 0.9, color="#2f7f6f", label="contamination removed")
axa.bar(x + w, tab.lostc, w * 0.9, color="#b5a642", label="own RNA removed")
axa.set_xticks(x, tab.label, rotation=40, ha="right", rotation_mode="anchor", fontsize=5.5)
axa.set_ylim(0, max(tab.other_share.max() * 100, tab.lostc.max()) * 1.45)
axa.set_ylabel("% of counts in cells")
axa.set_title("Removal by scCDC")
axa.grid(axis="y", color=GRID, lw=0.5)
axa.set_axisbelow(True)
axa.legend(fontsize=5, loc="upper left", handlelength=1, handletextpad=0.3, borderaxespad=0.1)
panel(axa, "a")

axb = fig.add_subplot(top[2])
is_g = curve.index.isin(curve_gcg)
axb.scatter(curve["mean.expr"][~is_g], curve.entropy[~is_g], s=0.6, color="#a9a8a3", lw=0, rasterized=True)
axb.scatter(curve["mean.expr"][is_g], curve.entropy[is_g], s=4, color="#c0392b", lw=0, label=f"GCGs ({is_g.sum()})")
o = np.argsort(curve["mean.expr"].to_numpy(), kind="stable")
axb.plot(curve["mean.expr"].to_numpy()[o], curve.fit.to_numpy()[o], color=INK, lw=0.8, label="fitted curve")
axb.set_xlabel("ln(mean count + 1)")
axb.set_ylabel("entropy of counts (bits)")
axb.set_title(f"Cluster {curve_name}, {LABEL[EXAMPLE].split(' (')[0]}")
axb.legend(fontsize=5, loc="upper left", handletextpad=0.2, borderaxespad=0.1, markerscale=2)
panel(axb, "b")

axc = fig.add_subplot(top[3])
lim, lo = removed_pairs.max() * 1.6, removed_pairs.min() / 1.6
axc.plot([lo, lim], [lo, lim], color=GRID, lw=0.6, zorder=0)
axc.scatter(removed_pairs[:, 0], removed_pairs[:, 1], s=14, color=C_R, edgecolor="white", linewidth=0.3,
            label="scCDC (R)")
axc.scatter(removed_pairs[:, 0], removed_pairs[:, 1], s=3, color=C_PY, lw=0, label="pysccdc")
axc.set_xscale("log")
axc.set_yscale("log")
axc.set_xlim(lo, lim)
axc.set_ylim(lo, lim)
axc.set_aspect("equal")
axc.set_xlabel("scCDC (R)")
axc.set_ylabel("pysccdc")
axc.set_title("Counts removed per GCG")
axc.text(0.96, 0.04, f"{len(removed_pairs)} GCGs\nidentical", transform=axc.transAxes, ha="right", va="bottom",
         fontsize=5, color=INK2)
panel(axc, "c")

bot = gs[1].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.3)
axd = fig.add_subplot(bot[0])
axd.scatter(tab.cells, tab.sec_R, s=12, color=C_R, edgecolor="white", linewidth=0.3, label="scCDC 1.4 (R)")
axd.scatter(tab.cells, tab.sec_py, s=12, color=C_PY, edgecolor="white", linewidth=0.3, label="pysccdc")
for _, r in tab.iterrows():
    axd.plot([r.cells] * 2, [r.sec_py, r.sec_R], color=GRID, lw=0.8, zorder=0)
axd.set_xscale("log")
axd.set_yscale("log")
axd.set_xlabel("cells")
axd.set_yticks([1, 3, 10, 30, 100, 300, 1000], ["1", "3", "10", "30", "100", "300", "1000"])
axd.minorticks_off()
axd.set_xticks([1000, 3000, 10000], ["1,000", "3,000", "10,000"])
axd.set_ylabel("wall time, 1 thread (s)")
axd.set_title("Run time, detection and correction")
axd.grid(color=GRID, lw=0.5)
axd.set_axisbelow(True)
axd.legend(fontsize=5.5, loc="upper left", handletextpad=0.1)
panel(axd, "d", dx=-0.2)

axe = fig.add_subplot(bot[1])
before, after_R = cellwise["R"]
_, after_py = cellwise["py"]
axe.scatter(before * 100, after_R * 100, s=1.5, color=C_R, lw=0, rasterized=True, label="scCDC (R)")
axe.scatter(before * 100, after_py * 100, s=0.4, color=C_PY, lw=0, rasterized=True, label="pysccdc")
top_ = np.nanpercentile(before * 100, 99.5)
axe.plot([0, top_], [0, top_], color=INK2, lw=0.6, zorder=0)
axe.set_xlim(0, top_)
axe.set_ylim(0, top_)
axe.set_xlabel("other-species counts before correction (% of cell)")
axe.set_ylabel("after correction (%)")
axe.set_title(f"Per cell, {LABEL[EXAMPLE]}")
axe.grid(color=GRID, lw=0.5)
axe.set_axisbelow(True)
leg = axe.legend(fontsize=5.5, loc="upper left", handletextpad=0.1)
for hh in leg.legend_handles:
    hh.set_sizes([10])
panel(axe, "e", dx=-0.2)

row3 = gs[2].subgridspec(1, 4, wspace=0.06)


def umap_ax(k, title, letter):
    ax = fig.add_subplot(row3[k])
    ax.set_xticks([])
    ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title(title)
    ax.set_aspect("equal")
    panel(ax, letter, dx=-0.04)
    return ax


ax = umap_ax(0, f"{LABEL[EXAMPLE]}: species", "f")
for name, col in [("human", "#2f7f6f"), ("mouse", "#b5a642"), ("mixed", "#2a2a2a")]:
    m = umap_species == name
    ax.scatter(xy[m, 0], xy[m, 1], s=0.6 if name != "mixed" else 1.2, c=col, lw=0, rasterized=True,
               label=f"{name} ({m.sum():,})")
leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=5.5, handletextpad=0.1,
                columnspacing=0.6)
for hh in leg.legend_handles:
    hh.set_sizes([10])
vmax = np.nanpercentile(umap_frac["before"] * 100, 99)
for k, (key, title, letter) in enumerate([("before", "Before correction", "g"), ("R", "After, scCDC (R)", "h"),
                                          ("py", "After, pysccdc", "i")]):
    ax = umap_ax(k + 1, title, letter)
    v = umap_frac[key] * 100
    nan = np.isnan(v)
    ax.scatter(xy[nan, 0], xy[nan, 1], s=0.6, c="#d9d8d3", lw=0, rasterized=True)
    o = np.flatnonzero(~nan)[np.argsort(v[~nan], kind="stable")]
    im = ax.scatter(xy[o, 0], xy[o, 1], c=v[o], s=0.6, cmap="magma_r", vmin=0, vmax=vmax, lw=0, rasterized=True)
cax = fig.axes[-2].inset_axes([0.05, -0.14, 0.9, 0.035])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=5.5, length=1.5)
cax.set_title("other-species counts (% of cell; mixed cells grey)", fontsize=5.5, color=INK2, pad=2)

pa, pd_ = axa.get_position(), axd.get_position()
x0, x1 = pd_.x0 - 0.2 * pd_.width, pa.x0 - 0.075
y0, y1 = pa.y0 - 0.02, pa.y1 + 0.035
h_ov = (x1 - x0) * 7.2 / 6.5 * logos[1].shape[0] / logos[1].shape[1]
for img, rect, anchor in [(logos[0], [x0, y0 + h_ov + 0.012, x1 - x0, y1 - y0 - h_ov - 0.012], "N"),
                          (logos[1], [x0, y0, x1 - x0, h_ov], "S")]:
    axl = fig.add_axes(rect)
    axl.imshow(img)
    axl.set_anchor(anchor)
    axl.set_axis_off()
fig.savefig(OUT / "fig1.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "fig1.png", bbox_inches="tight", pad_inches=0.02, dpi=200)

# ======================= numbers =======================
speed = tab.sec_R / tab.sec_py
big = tab.iloc[tab.cells.idxmax()]
ex = tab.set_index("dataset").loc[EXAMPLE]
macros = {
    "nDatasets": len(tab), "nCellsTotal": f"{int(tab.cells.sum()):,}",
    "nCellsMin": f"{int(tab.cells.min()):,}", "nCellsMax": f"{int(tab.cells.max()):,}",
    "nSameGcg": int(tab.same_gcg.sum()), "nGcgTotal": len(removed_pairs),
    "gcgMin": int(tab.n_gcg_R.min()), "gcgMax": int(tab.n_gcg_R.max()),
    "distMaxDiff": f"{tab.dist_maxdiff.max():.0e}", "corrMaxDiff": f"{tab.corr_maxdiff.max():g}",
    "otherMin": f"{tab.other_share.min() * 100:.1f}", "otherMax": f"{tab.other_share.max() * 100:.1f}",
    "remMin": f"{tab.removed_R.min() * 100:.1f}", "remMax": f"{tab.removed_R.max() * 100:.1f}",
    "lostMin": f"{tab.lost_R.min() * 100:.2f}", "lostMax": f"{tab.lost_R.max() * 100:.2f}",
    "remcMax": f"{tab.remc.max():.2f}", "lostcMax": f"{tab.lostc.max():.2f}",
    "speedMin": f"{speed.min():.0f}", "speedMax": f"{speed.max():.0f}",
    "tBigR": f"{big.sec_R:.0f}", "tBigPy": f"{big.sec_py:.0f}", "bigName": big.label,
    "exampleName": LABEL[EXAMPLE], "exCluster": curve_name, "exClusterGcg": int(is_g.sum()),
    "nFailed": len(failed), "failedName": ", ".join(LABEL[f] for f in failed),
}
for key, name in [("OneK", "hgmm_1k_v3"), ("FiveK", "hgmm_5k_v3"), ("SixK", "hgmm_6k"), ("TwelveK", "hgmm_12k")]:
    if name in set(tab.dataset):
        r_ = tab.set_index("dataset").loc[name]
        macros[f"gcg{key}"] = int(r_.n_gcg_R)
        macros[f"rem{key}"] = f"{r_.removed_R * 100:.1f}"
        macros[f"lost{key}"] = f"{r_.lost_R * 100:.2f}"
with open(OUT / "numbers.tex", "w") as fh:
    for k, v in macros.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(open(OUT / "numbers.tex").read())
print(tab.drop(columns=["label"]).round(5).T.to_string())
