import os, numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdShapeHelpers, rdMolAlign
from rdkit.Chem.FeatMaps import FeatMaps
from rdkit import RDConfig
import matplotlib.pyplot as plt
from collections import namedtuple
import pandas as pd
import csv

# ---------- config ----------
REF_SDF   = "data/similarity_experiments/sucos_sim/sucos_aripip_lr1e-5_ema_mean_experiment/epoch_0000/epoch0_data_drd2_strucutres_processed_ligand_free_pockets_drd2_7e2z_E_9SC_pocket_only.pdb_data_drd2_strucutres_7e2z_E_9SC_lig0_ref.sdf"  # single-mol SDF of the reference pose
DIFF_SDF  = "data/qed_sigmoid/DiffSBDD_test_pockets/DiffSBDD_7e2z_30_nodes.sdf"                                            # 100 mols
PRISM_SDF = "data/similarity_experiments/sucos_sim/test_time_last_ckpt_gen_100_30nodes/SuCOS_7e2z_centered_lr_1e-5_clip_0.1_30_nodes.sdf"  # 100 mols
OUT_CSV   = "sucos_comparison_7e2z_diffsbdd_prism.csv"
ADD_HS    = True
ALLOW_REORDERING = False
WEIGHT_SHAPE = 0.5
WEIGHT_FEATS = 0.5

# Consistent colors
COLOR_PRISM = "orange"
COLOR_DIFF  = "steelblue"
# ----------------------------

# Feature factory
fdef = AllChem.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
KEEP = ('Donor','Acceptor','NegIonizable','PosIonizable','ZnBinder','Aromatic','Hydrophobe','LumpedHydrophobe')
fmParams = {k: FeatMaps.FeatMapParams() for k in fdef.GetFeatureFamilies()}

def load_single_mol(path):
    m = Chem.MolFromMolFile(path, removeHs=not ADD_HS)
    if m is None:
        supp = Chem.SDMolSupplier(path, removeHs=not ADD_HS)
        for x in supp:
            if x is not None:
                m = x
                break
    if m is None:
        raise ValueError(f"Could not read molecule from {path}")
    if ADD_HS: m = Chem.AddHs(m, addCoords=True)
    if m.GetNumConformers() == 0:
        AllChem.EmbedMolecule(m, randomSeed=0xf00d)
    return m

def load_many(path):
    supp = Chem.SDMolSupplier(path, removeHs=not ADD_HS)
    mols = [Chem.AddHs(m, addCoords=True) if (m is not None and ADD_HS) else m for m in supp if m is not None]
    for m in mols:
        if m.GetNumConformers() == 0:
            AllChem.EmbedMolecule(m, randomSeed=0xbeef)
    return mols

def feature_map_score(ref, prb):
    feats_ref = [f for f in fdef.GetFeaturesForMol(ref) if f.GetFamily() in KEEP]
    feats_prb = [f for f in fdef.GetFeaturesForMol(prb) if f.GetFamily() in KEEP]
    fm_ref = FeatMaps.FeatMap(feats=feats_ref, weights=[1]*len(feats_ref), params=fmParams)
    fm_ref.scoreMode = FeatMaps.FeatMapScoreMode.Best   # normalized by min(n_ref, n_prb)
    score = fm_ref.ScoreFeats(feats_prb)
    denom = min(fm_ref.GetNumFeatures(), len(feats_prb)) or 1
    return float(score / denom)

def sucos_components(ref, prb):
    try:
        rdMolAlign.AlignMol(prb, ref)
    except Exception:
        pass
    protrude = rdShapeHelpers.ShapeProtrudeDist(ref, prb, allowReordering=ALLOW_REORDERING)
    fm = feature_map_score(ref, prb)
    return fm, 1.0 - float(protrude)

def score_set(ref, mols, tag):
    rows = []
    for i, m in enumerate(mols):
        if m is None:
            continue
        try:
            fm, shape = sucos_components(ref, m)
            sucos = WEIGHT_FEATS*fm + WEIGHT_SHAPE*shape
            rows.append((tag, i, sucos, fm, shape))
        except Exception:
            rows.append((tag, i, np.nan, np.nan, np.nan))
    return rows

# Load
ref = load_single_mol(REF_SDF)
diff_mols  = load_many(DIFF_SDF)
prism_mols = load_many(PRISM_SDF)

# Score
rows = []
rows += score_set(ref, prism_mols, "PRISM")      # PRISM first
rows += score_set(ref, diff_mols,  "DiffSBDD")   # baseline second

# Save CSV
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["set","idx","sucos","features","shape"])
    for r in rows:
        w.writerow(r)

# Summaries
def summarize(vals):
    vals = np.array([v for v in vals if np.isfinite(v)])
    return dict(n=len(vals), mean=float(vals.mean()), median=float(np.median(vals)),
                q25=float(np.quantile(vals,0.25)), q75=float(np.quantile(vals,0.75)))

df = pd.DataFrame(rows, columns=["set","idx","sucos","features","shape"])

# Arrays, PRISM first
sucos_prism  = df[df["set"]=="PRISM"]["sucos"].dropna().values
sucos_diff   = df[df["set"]=="DiffSBDD"]["sucos"].dropna().values
fm_prism     = df[df["set"]=="PRISM"]["features"].dropna().values
fm_diff      = df[df["set"]=="DiffSBDD"]["features"].dropna().values
shape_prism  = df[df["set"]=="PRISM"]["shape"].dropna().values
shape_diff   = df[df["set"]=="DiffSBDD"]["shape"].dropna().values

print("SuCOS summary")
print("PRISM   :", summarize(sucos_prism))
print("DiffSBDD:", summarize(sucos_diff))

# Success fractions
for thr in [0.40, 0.50]:
    p = np.mean(sucos_prism >= thr) if len(sucos_prism) else np.nan
    d = np.mean(sucos_diff  >= thr) if len(sucos_diff)  else np.nan
    print(f"Frac ≥ {thr:.2f}  PRISM={p:.3f}  DiffSBDD={d:.3f}  Δ={p-d:+.3f}")

# ========== Mann–Whitney U + Cliff's delta ==========
def _as_clean_array(x):
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]

def cliffs_delta(x, y):
    """
    Cliff's delta: P(X>Y) - P(X<Y).
    |delta| thresholds: ~0.147 small, 0.33 medium, 0.474 large.
    """
    x = _as_clean_array(x); y = _as_clean_array(y)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0: return np.nan
    # efficient pairwise compare up to ~1e5 pairs; else sample
    if nx * ny <= 100000:
        diff = x[:, None] - y[None, :]
        return (np.sum(diff > 0) - np.sum(diff < 0)) / (nx * ny)
    rng = np.random.default_rng(12345)
    xs = rng.choice(x, size=min(nx, 500), replace=False)
    ys = rng.choice(y, size=min(ny, 500), replace=False)
    diff = xs[:, None] - ys[None, :]
    return (np.sum(diff > 0) - np.sum(diff < 0)) / (xs.size * ys.size)

def mw_and_delta(x, y, label):
    print(f"\n=== {label}: PRISM vs DiffSBDD ===")
    x = _as_clean_array(x); y = _as_clean_array(y)
    print(f"n_PRISM={len(x)}, n_DiffSBDD={len(y)}")
    try:
        from scipy.stats import mannwhitneyu
        u = mannwhitneyu(x, y, alternative="two-sided")
        print(f"Mann–Whitney U: statistic={u.statistic:.4g}, p={u.pvalue:.3g}")
    except Exception as e:
        print(f"(Note) SciPy not available or error '{e}'. Could not run Mann–Whitney.")
        u = None
    delta = cliffs_delta(x, y)
    print(f"Cliff's delta: {delta:.3f}  (|δ|≈0.147 small, 0.33 med, 0.474 large)")
    return u, delta

u_sucos,  d_sucos  = mw_and_delta(sucos_prism, sucos_diff, "SuCOS")
u_feat,   d_feat   = mw_and_delta(fm_prism, fm_diff, "Feature score")
u_shape,  d_shape  = mw_and_delta(shape_prism, shape_diff, "Shape overlap")

# ---------- Plots ----------
# 1) Overall SuCOS: histogram + violin, PRISM first and colored
def two_hist(ax, prism_vals, diff_vals, bins=20):
    ax.hist(prism_vals, bins=bins, alpha=0.6, color=COLOR_PRISM, label="PRISM")
    ax.hist(diff_vals,  bins=bins, alpha=0.6, color=COLOR_DIFF,  label="DiffSBDD")
    ax.set_xlabel("SuCOS")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)

def violin(ax, data, labels, ylabel):
    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    # color the violins
    for pc, color in zip(parts['bodies'], [COLOR_PRISM, COLOR_DIFF]):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    ax.set_xticks(range(1, len(labels)+1))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)

fig, axes = plt.subplots(1, 2, figsize=(10,4))
two_hist(axes[0], sucos_prism, sucos_diff, bins=20)
violin(axes[1], [sucos_prism, sucos_diff], ["PRISM", "DiffSBDD"], ylabel="SuCOS")

# Attempt to annotate SuCOS violin with p and delta
def add_text_annotation(ax, u_res, delta, y_frac=0.95):
    txt_lines = []
    if u_res is not None:
        txt_lines.append(f"MW p={u_res.pvalue:.3g}")
    txt_lines.append(f"Cliff δ={delta:.3f}")
    txt = "\n".join(txt_lines)
    ymin, ymax = ax.get_ylim()
    ax.text(1.02, ymin + y_frac*(ymax-ymin), txt, ha='left', va='top', transform=ax.get_yaxis_transform())

# If statannotations is available, use it for SuCOS; else write text
try:
    from statannotations.Annotator import Annotator
    data_sucos = pd.DataFrame({
        "set": (["PRISM"]*len(sucos_prism)) + (["DiffSBDD"]*len(sucos_diff)),
        "value": np.r_[sucos_prism, sucos_diff],
    })
    # build a fresh axis for clean annotation
    fig_ann, ax_ann = plt.subplots(1,1, figsize=(5,4))
    parts = ax_ann.violinplot([sucos_prism, sucos_diff], showmeans=True, showmedians=True)
    for pc, color in zip(parts['bodies'], [COLOR_PRISM, COLOR_DIFF]):
        pc.set_facecolor(color); pc.set_alpha(0.6)
    ax_ann.set_xticks([1,2]); ax_ann.set_xticklabels(["PRISM","DiffSBDD"])
    ax_ann.set_ylabel("SuCOS")
    annot = Annotator(ax_ann, [("PRISM", "DiffSBDD")], data=data_sucos, x="set", y="value", order=["PRISM","DiffSBDD"])
    annot.configure(test='Mann-Whitney', text_format='simple', loc='inside', show_test_name=False)
    annot.apply_and_annotate()
    plt.tight_layout()
    plt.show()
except Exception:
    add_text_annotation(axes[1], u_sucos, d_sucos)

plt.tight_layout()
plt.show()

# 2) Component violins, PRISM first, consistent colors + text annotations
fig, axes = plt.subplots(1, 2, figsize=(10,4))

parts = axes[0].violinplot([fm_prism, fm_diff], showmeans=True, showmedians=True)
for pc, color in zip(parts['bodies'], [COLOR_PRISM, COLOR_DIFF]):
    pc.set_facecolor(color); pc.set_alpha(0.6)
axes[0].set_xticks([1,2]); axes[0].set_xticklabels(["PRISM","DiffSBDD"])
axes[0].set_ylabel("Feature Score")
add_text_annotation(axes[0], u_feat, d_feat)

parts = axes[1].violinplot([shape_prism, shape_diff], showmeans=True, showmedians=True)
for pc, color in zip(parts['bodies'], [COLOR_PRISM, COLOR_DIFF]):
    pc.set_facecolor(color); pc.set_alpha(0.6)
axes[1].set_xticks([1,2]); axes[1].set_xticklabels(["PRISM","DiffSBDD"])
axes[1].set_ylabel("Shape Overlap")
add_text_annotation(axes[1], u_shape, d_shape)

plt.tight_layout()
plt.show()
