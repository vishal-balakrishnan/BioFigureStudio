import io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from statsmodels.stats.multitest import multipletests

st.set_page_config(page_title="BioFigureStudio v3", layout="wide")
st.title("BioFigureStudio v3")
st.caption("Raw-matrix statistics, clustered heatmaps, Venn overlaps & 300/600-DPI vector export")


# ==============================================================================
# CORE OOP ENGINE (Built with a Python Class as recommended by Dr. Rao)
# ==============================================================================
class BioPipeline:
    """Encapsulates data cleaning, QC auditing, and differential statistics."""

    def __init__(self, raw_df: pd.DataFrame, gene_col: str = "Gene"):
        self.raw_df = raw_df.copy()
        self.gene_col = gene_col
        self.clean_df = pd.DataFrame()
        self.audit = {}

    def clean(self, value_cols: list[str]) -> pd.DataFrame:
        df = self.raw_df.copy()
        if self.gene_col != "Gene":
            df = df.rename(columns={self.gene_col: "Gene"})

        raw_rows = len(df)
        df = df.dropna(subset=["Gene"] + value_cols)
        dropped_rows = raw_rows - len(df)

        raw_genes = df["Gene"].astype(str)
        clean_genes = raw_genes.str.strip().str.upper()
        names_fixed = int((raw_genes != clean_genes).sum())
        df["Gene"] = clean_genes

        before_dedup = len(df)
        df = df.drop_duplicates(subset=["Gene"], keep="first").reset_index(drop=True)
        dupes_removed = before_dedup - len(df)

        for col in value_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=value_cols).reset_index(drop=True)

        self.clean_df = df
        self.audit = {
            "raw_rows": raw_rows,
            "dropped_rows": dropped_rows,
            "names_fixed": names_fixed,
            "dupes_removed": dupes_removed,
            "final_rows": len(df),
        }
        return self.clean_df

    def compute_differential_stats(
        self,
        group_a_cols: list[str],
        group_b_cols: list[str],
        adj_method: str = "fdr_bh",
        fc_cutoff: float = 1.0,
        padj_cutoff: float = 0.05,
    ) -> pd.DataFrame:
        df = self.clean_df.copy()
        a_vals = df[group_a_cols].to_numpy(dtype=float)
        b_vals = df[group_b_cols].to_numpy(dtype=float)

        # Assumes log2-normalized expression values; falls back to log2 ratio if positive raw counts
        mean_a = np.mean(a_vals, axis=1)
        mean_b = np.mean(b_vals, axis=1)
        if np.min(a_vals) >= 0 and np.min(b_vals) >= 0 and np.max(a_vals) > 50:
            log2fc = np.log2((mean_b + 1.0) / (mean_a + 1.0))
        else:
            log2fc = mean_b - mean_a

        _, p_vals = stats.ttest_ind(b_vals, a_vals, axis=1, equal_var=False)
        p_vals = np.nan_to_num(p_vals, nan=1.0)
        p_vals = np.clip(p_vals, 1e-300, 1.0)

        _, adj_p_vals, _, _ = multipletests(p_vals, method=adj_method)
        adj_p_vals = np.clip(adj_p_vals, 1e-300, 1.0)

        df["log2FC"] = np.round(log2fc, 3)
        df["p_value"] = p_vals
        df["adj_p_value"] = adj_p_vals
        df["neg_log10_padj"] = -np.log10(adj_p_vals)

        df["Status"] = "Not Significant"
        df.loc[(df["log2FC"] >= fc_cutoff) & (df["adj_p_value"] < padj_cutoff), "Status"] = "Upregulated"
        df.loc[(df["log2FC"] <= -fc_cutoff) & (df["adj_p_value"] < padj_cutoff), "Status"] = "Downregulated"
        return df


# ==============================================================================
# SAMPLE RAW MULTI-GROUP DATASET (3 Groups x 4 Replicates = 12 Samples)
# ==============================================================================
def make_sample_matrix() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    genes = [
        " blaNDM-1 ", " mecA", "vanA ", " blaKPC-2", "ermB ", " tetM", "ctx-M-15 ", "aac(6')-Ib",
        " sul1 ", "dfrA1 ", " ompK36 ", "porB ", " acrR", "marR ", " gyrA_wt", "pmrB ",
    ] + [f"gene_{i:02d}" for i in range(1, 25)]

    n = len(genes)
    ctrl = rng.normal(6.0, 0.35, (n, 4))
    trt_a = rng.normal(6.0, 0.35, (n, 4))
    trt_b = rng.normal(6.0, 0.35, (n, 4))

    # Inject strong biological signal into first 16 AMR/stress genes
    trt_a[:8] += rng.uniform(2.0, 3.8, (8, 1))
    trt_a[10:15] -= rng.uniform(1.8, 3.2, (5, 1))

    trt_b[2:10] += rng.uniform(1.9, 3.5, (8, 1))
    trt_b[12:16] -= rng.uniform(2.0, 3.4, (4, 1))

    data = {"Gene": genes}
    for i in range(4):
        data[f"Control_{i+1}"] = np.round(ctrl[:, i], 3)
    for i in range(4):
        data[f"TreatedA_{i+1}"] = np.round(trt_a[:, i], 3)
    for i in range(4):
        data[f"TreatedB_{i+1}"] = np.round(trt_b[:, i], 3)
    return pd.DataFrame(data)


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
st.sidebar.header("1. Raw Sample Matrix Input")
use_sample = st.sidebar.button("Load Sample 3-Group Matrix", type="primary")
uploaded_file = st.sidebar.file_uploader("Or upload raw CSV (Gene + Sample columns)", type=["csv"])

if "raw_matrix" not in st.session_state:
    st.session_state.raw_matrix = make_sample_matrix()

if use_sample:
    st.session_state.raw_matrix = make_sample_matrix()
elif uploaded_file is not None:
    uploaded_file.seek(0)
    st.session_state.raw_matrix = pd.read_csv(uploaded_file)

raw_df = st.session_state.raw_matrix.copy()
all_cols = list(raw_df.columns)
default_gene_col = all_cols[0]
numeric_cols = [c for c in all_cols if c != default_gene_col]

st.sidebar.header("2. Group & Stats Setup")
gene_col = st.sidebar.selectbox("Gene ID Column", all_cols, index=0)
sample_cols = [c for c in all_cols if c != gene_col]

# Auto-detect 3 sample groups when using default naming, else let user pick
g1_default = [c for c in sample_cols if "Control" in c] or sample_cols[:2]
g2_default = [c for c in sample_cols if "TreatedA" in c] or sample_cols[2:4]
g3_default = [c for c in sample_cols if "TreatedB" in c] or []

grp_ctrl = st.sidebar.multiselect("Group 1 (Control)", sample_cols, default=g1_default)
grp_a = st.sidebar.multiselect("Group 2 (Treated A)", sample_cols, default=g2_default)
grp_b = st.sidebar.multiselect("Group 3 (Treated B, optional)", sample_cols, default=g3_default)

adj_choice = st.sidebar.selectbox(
    "Multiple-Testing Correction (Adjusted p-value)",
    ["Benjamini-Hochberg (FDR)", "Bonferroni", "Holm-Sidak"],
)
adj_map = {"Benjamini-Hochberg (FDR)": "fdr_bh", "Bonferroni": "bonferroni", "Holm-Sidak": "holm-sidak"}
adj_method = adj_map[adj_choice]

fc_cutoff = st.sidebar.slider("|log2 Fold Change| cutoff", 0.5, 3.0, 1.0, 0.1)
padj_cutoff = st.sidebar.slider("Adjusted p-value cutoff", 0.001, 0.10, 0.05, 0.005, format="%.3f")

st.sidebar.header("3. Journal Export Presets")
layout_choice = st.sidebar.selectbox("Figure Size", ["Double-Column (6.5 x 4.8 in)", "Single-Column (3.5 x 3.2 in)"])
export_dpi = st.sidebar.radio("Resolution (DPI)", [300, 600], horizontal=True)
top_n = st.sidebar.slider("Label top N genes", 0, 15, 8)
fig_size = (6.5, 4.8) if "Double" in layout_choice else (3.5, 3.2)

# Validate selected groups
active_groups = {"Control": grp_ctrl, "Treated A": grp_a}
if grp_b:
    active_groups["Treated B"] = grp_b

if len(grp_ctrl) < 2 or len(grp_a) < 2:
    st.warning("Select at least 2 sample columns for Group 1 and Group 2 to compute t-test p-values.")
    st.stop()

used_sample_cols = list(dict.fromkeys(grp_ctrl + grp_a + grp_b))

# ==============================================================================
# RUN OOP PIPELINE
# ==============================================================================
pipeline = BioPipeline(raw_df, gene_col=gene_col)
clean_df = pipeline.clean(used_sample_cols)
audit = pipeline.audit

# Pairwise comparison radio buttons (Dr. Rao's multi-group request)
pairs = [("Treated A vs Control", grp_ctrl, grp_a)]
if len(grp_b) >= 2:
    pairs.append(("Treated B vs Control", grp_ctrl, grp_b))
    pairs.append(("Treated B vs Treated A", grp_a, grp_b))

pair_labels = [p[0] for p in pairs]
selected_pair = st.radio("Pairwise Comparison", pair_labels, horizontal=True)
pair_dict = {label: (g1, g2) for label, g1, g2 in pairs}
base_cols, comp_cols = pair_dict[selected_pair]

res_df = pipeline.compute_differential_stats(
    base_cols, comp_cols, adj_method=adj_method, fc_cutoff=fc_cutoff, padj_cutoff=padj_cutoff
)
sig_df = res_df[res_df["Status"] != "Not Significant"].sort_values("adj_p_value")

# Audit Log & Top KPIs
with st.expander("Data Cleaning Audit Log (QC Report)", expanded=True):
    st.write(
        f"**Raw rows:** {audit['raw_rows']} | "
        f"**Empty/invalid rows dropped:** {audit['dropped_rows']} | "
        f"**Gene IDs standardized:** {audit['names_fixed']} | "
        f"**Duplicate genes removed:** {audit['dupes_removed']} | "
        f"**Clean genes analyzed:** {audit['final_rows']}"
    )

c1, c2, c3, c4 = st.columns(4)
c1.metric("Clean Genes", len(res_df))
c2.metric("Upregulated", int((res_df["Status"] == "Upregulated").sum()))
c3.metric("Downregulated", int((res_df["Status"] == "Downregulated").sum()))
c4.metric("Correction", adj_choice.split(" ")[0])

# ==============================================================================
# 3 PUBLICATION PLOT TABS
# ==============================================================================
tab1, tab2, tab3 = st.tabs(["1. Volcano Plot (Adj. p-value)", "2. Clustered Heatmap (Tree Cutoffs)", "3. Venn Overlap (Counts + %)"])

# --- TAB 1: VOLCANO PLOT ---
with tab1:
    col_plot, col_table = st.columns([1.2, 1.0])
    with col_plot:
        st.subheader(f"Volcano Plot: {selected_pair} ({export_dpi}-DPI)")
        fig_v, ax_v = plt.subplots(figsize=fig_size, dpi=export_dpi)
        palette = {"Upregulated": "#d97706", "Downregulated": "#0f4c5c", "Not Significant": "#cbd5e1"}

        for status, grp in res_df.groupby("Status"):
            ax_v.scatter(
                grp["log2FC"], grp["neg_log10_padj"],
                c=palette[status], label=f"{status} ({len(grp)})",
                s=42 if status != "Not Significant" else 24,
                alpha=0.88, edgecolors="white", linewidth=0.5,
            )

        ax_v.axvline(fc_cutoff, color="#475569", linestyle="--", linewidth=1)
        ax_v.axvline(-fc_cutoff, color="#475569", linestyle="--", linewidth=1)
        ax_v.axhline(-np.log10(padj_cutoff), color="#475569", linestyle="--", linewidth=1)

        for _, row in sig_df.head(top_n).iterrows():
            ax_v.annotate(row["Gene"], (row["log2FC"], row["neg_log10_padj"]),
                          xytext=(4, 4), textcoords="offset points", fontsize=8)

        ax_v.set_xlabel("log2 Fold Change")
        ax_v.set_ylabel(f"-log10(Adjusted p-value: {adj_choice.split(' ')[0]})")
        ax_v.set_title(f"{selected_pair} ({adj_choice})")
        ax_v.legend(frameon=False, fontsize=8, loc="upper left")
        ax_v.spines["top"].set_visible(False)
        ax_v.spines["right"].set_visible(False)
        st.pyplot(fig_v)

    with col_table:
        st.subheader("Significant Hits (Ranked by Adj. p-value)")
        st.dataframe(sig_df[["Gene", "log2FC", "p_value", "adj_p_value", "Status"]], use_container_width=True, height=340)

    pdf_v, svg_v = io.BytesIO(), io.BytesIO()
    fig_v.savefig(pdf_v, format="pdf", dpi=export_dpi, bbox_inches="tight")
    fig_v.savefig(svg_v, format="svg", dpi=export_dpi, bbox_inches="tight")
    plt.close(fig_v)

    d1, d2, d3 = st.columns(3)
    d1.download_button(f"Download Volcano {export_dpi}-DPI PDF", pdf_v.getvalue(), f"volcano_{export_dpi}dpi.pdf", "application/pdf")
    d2.download_button("Download Volcano SVG", svg_v.getvalue(), "volcano.svg", "image/svg+xml")
    d3.download_button("Download Stats CSV", res_df[["Gene", "log2FC", "p_value", "adj_p_value", "Status"]].to_csv(index=False), "differential_stats.csv", "text/csv")

# --- TAB 2: CLUSTERED HEATMAP WITH TREE CUTOFF GAPS ---
with tab2:
    st.subheader("Clustered Heatmap (pheatmap-style with Tree Cutoff Gaps)")
    hm_col1, hm_col2 = st.columns([1, 2])
    with hm_col1:
        n_hm_genes = st.slider("Top genes to cluster", 10, min(40, len(res_df)), min(20, len(res_df)))
        k_clusters = st.radio("Row tree cutoff (clusters)", [2, 3, 4], horizontal=True)

    hm_df = res_df.sort_values("adj_p_value").head(n_hm_genes).set_index("Gene")[used_sample_cols]
    # Row Z-score standardization so genes compare on relative expression
    row_means = hm_df.mean(axis=1)
    row_stds = hm_df.std(axis=1).replace(0, 1.0)
    z_df = hm_df.sub(row_means, axis=0).div(row_stds, axis=0)

    row_link = linkage(z_df.to_numpy(), method="ward")
    col_link = linkage(z_df.to_numpy().T, method="ward")
    row_dendro = dendrogram(row_link, no_plot=True)
    col_dendro = dendrogram(col_link, no_plot=True)

    row_order = row_dendro["leaves"]
    col_order = col_dendro["leaves"]
    z_ordered = z_df.iloc[row_order, col_order]
    clusters = fcluster(row_link, t=k_clusters, criterion="maxclust")[row_order]

    # Build figure with dendrogram + cluster-split heatmap blocks
    fig_h = plt.figure(figsize=(8.0, 6.2), dpi=export_dpi)
    gs = fig_h.add_gridspec(2, 2, width_ratios=[0.18, 0.82], height_ratios=[0.15, 0.85], wspace=0.03, hspace=0.03)

    ax_cd = fig_h.add_subplot(gs[0, 1])
    dendrogram(col_link, ax=ax_cd, color_threshold=0, above_threshold_color="#334155")
    ax_cd.axis("off")

    ax_rd = fig_h.add_subplot(gs[1, 0])
    dendrogram(row_link, ax=ax_rd, orientation="left", color_threshold=0, above_threshold_color="#334155")
    ax_rd.invert_yaxis()
    ax_rd.axis("off")

    ax_hm = fig_h.add_subplot(gs[1, 1])
    mat = z_ordered.to_numpy()
    vmin, vmax = -2.0, 2.0
    cmap = plt.get_cmap("RdBu_r")

    # Draw cells with white gap rows wherever cluster ID changes (pheatmap cutree_rows look)
    y_pos = 0.0
    y_ticks, y_labels = [], []
    for r_idx in range(len(z_ordered)):
        if r_idx > 0 and clusters[r_idx] != clusters[r_idx - 1]:
            y_pos += 0.45  # crisp cluster gap
        for c_idx in range(len(z_ordered.columns)):
            val = np.clip((mat[r_idx, c_idx] - vmin) / (vmax - vmin), 0.0, 1.0)
            rect = patches.Rectangle((c_idx, y_pos), 1, 1, facecolor=cmap(val), edgecolor="white", linewidth=0.6)
            ax_hm.add_patch(rect)
        y_ticks.append(y_pos + 0.5)
        y_labels.append(z_ordered.index[r_idx])
        y_pos += 1.0

    ax_hm.set_xlim(0, len(z_ordered.columns))
    ax_hm.set_ylim(0, y_pos)
    ax_hm.invert_yaxis()
    ax_hm.set_xticks(np.arange(len(z_ordered.columns)) + 0.5)
    ax_hm.set_xticklabels(z_ordered.columns, rotation=45, ha="right", fontsize=8)
    ax_hm.set_yticks(y_ticks)
    ax_hm.set_yticklabels(y_labels, fontsize=8)
    ax_hm.yaxis.tick_right()
    for spine in ax_hm.spines.values():
        spine.set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig_h.colorbar(sm, ax=ax_hm, fraction=0.03, pad=0.16, label="Row Z-Score")
    st.pyplot(fig_h)

    pdf_h, svg_h = io.BytesIO(), io.BytesIO()
    fig_h.savefig(pdf_h, format="pdf", dpi=export_dpi, bbox_inches="tight")
    fig_h.savefig(svg_h, format="svg", dpi=export_dpi, bbox_inches="tight")
    plt.close(fig_h)

    h1, h2 = st.columns(2)
    h1.download_button(f"Download Heatmap {export_dpi}-DPI PDF", pdf_h.getvalue(), f"heatmap_{export_dpi}dpi.pdf", "application/pdf")
    h2.download_button("Download Heatmap SVG", svg_h.getvalue(), "heatmap.svg", "image/svg+xml")

# --- TAB 3: VENN DIAGRAM (COUNTS + PERCENTAGES) ---
with tab3:
    st.subheader("Significant Gene Overlap (Counts & Percentages)")
    set_dict = {}
    for label, g1, g2 in pairs:
        s_df = pipeline.compute_differential_stats(g1, g2, adj_method=adj_method, fc_cutoff=fc_cutoff, padj_cutoff=padj_cutoff)
        set_dict[label] = set(s_df[s_df["Status"] != "Not Significant"]["Gene"])

    if len(pairs) < 2:
        st.info("Select at least 2 sample columns in Group 3 (Treated B) in the sidebar to compare 2–3 gene sets in a Venn diagram.")
    else:
        fig_vn, ax_vn = plt.subplots(figsize=(6.2, 4.8), dpi=export_dpi)
        ax_vn.set_aspect("equal")
        ax_vn.axis("off")

        names = list(set_dict.keys())
        sets = [set_dict[n] for n in names]
        union_all = set().union(*sets)
        total_u = max(len(union_all), 1)

        def fmt(n_val: int) -> str:
            return f"{n_val}\n({100 * n_val / total_u:.1f}%)"

        if len(sets) == 2:
            s1, s2 = sets
            only1 = len(s1 - s2)
            only2 = len(s2 - s1)
            both = len(s1 & s2)
            c1_p = patches.Circle((-0.55, 0), 1.0, facecolor="#0f4c5c", alpha=0.35, edgecolor="#0f4c5c", linewidth=2)
            c2_p = patches.Circle((0.55, 0), 1.0, facecolor="#d97706", alpha=0.35, edgecolor="#d97706", linewidth=2)
            ax_vn.add_patch(c1_p)
            ax_vn.add_patch(c2_p)
            ax_vn.text(-1.0, 0, fmt(only1), ha="center", va="center", fontsize=10, fontweight="bold")
            ax_vn.text(1.0, 0, fmt(only2), ha="center", va="center", fontsize=10, fontweight="bold")
            ax_vn.text(0.0, 0, fmt(both), ha="center", va="center", fontsize=10, fontweight="bold")
            ax_vn.text(-0.7, 1.15, names[0], ha="center", fontsize=9, fontweight="bold", color="#0f4c5c")
            ax_vn.text(0.7, 1.15, names[1], ha="center", fontsize=9, fontweight="bold", color="#d97706")
            ax_vn.set_xlim(-1.8, 1.8)
            ax_vn.set_ylim(-1.3, 1.4)
        else:
            s1, s2, s3 = sets[0], sets[1], sets[2]
            coords = [(-0.55, 0.35, "#0f4c5c"), (0.55, 0.35, "#d97706"), (0.0, -0.55, "#2563eb")]
            for cx, cy, col in coords:
                ax_vn.add_patch(patches.Circle((cx, cy), 0.95, facecolor=col, alpha=0.28, edgecolor=col, linewidth=2))

            ax_vn.text(-0.95, 0.65, fmt(len(s1 - s2 - s3)), ha="center", va="center", fontsize=9, fontweight="bold")
            ax_vn.text(0.95, 0.65, fmt(len(s2 - s1 - s3)), ha="center", va="center", fontsize=9, fontweight="bold")
            ax_vn.text(0.0, -0.95, fmt(len(s3 - s1 - s2)), ha="center", va="center", fontsize=9, fontweight="bold")
            ax_vn.text(0.0, 0.62, fmt(len((s1 & s2) - s3)), ha="center", va="center", fontsize=9, fontweight="bold")
            ax_vn.text(-0.55, -0.25, fmt(len((s1 & s3) - s2)), ha="center", va="center", fontsize=9, fontweight="bold")
            ax_vn.text(0.55, -0.25, fmt(len((s2 & s3) - s1)), ha="center", va="center", fontsize=9, fontweight="bold")
            ax_vn.text(0.0, 0.12, fmt(len(s1 & s2 & s3)), ha="center", va="center", fontsize=9, fontweight="bold")

            ax_vn.text(-0.85, 1.42, names[0], ha="center", fontsize=8.5, fontweight="bold", color="#0f4c5c")
            ax_vn.text(0.85, 1.42, names[1], ha="center", fontsize=8.5, fontweight="bold", color="#d97706")
            ax_vn.text(0.0, -1.62, names[2], ha="center", fontsize=8.5, fontweight="bold", color="#2563eb")
            ax_vn.set_xlim(-1.8, 1.8)
            ax_vn.set_ylim(-1.8, 1.65)

        ax_vn.set_title(f"Significant Gene Overlap (Total Union = {len(union_all)} genes)", fontsize=10, fontweight="bold")
        st.pyplot(fig_vn)

        pdf_vn, svg_vn = io.BytesIO(), io.BytesIO()
        fig_vn.savefig(pdf_vn, format="pdf", dpi=export_dpi, bbox_inches="tight")
        fig_vn.savefig(svg_vn, format="svg", dpi=export_dpi, bbox_inches="tight")
        plt.close(fig_vn)

        v1, v2 = st.columns(2)
        v1.download_button(f"Download Venn {export_dpi}-DPI PDF", pdf_vn.getvalue(), f"venn_{export_dpi}dpi.pdf", "application/pdf")
        v2.download_button("Download Venn SVG", svg_vn.getvalue(), "venn.svg", "image/svg+xml")

# ==============================================================================
# AUTO-GENERATED METHODS SUMMARY
# ==============================================================================
st.subheader("Auto-Generated Materials & Methods")
st.info(
    f"Raw expression matrices ({audit['raw_rows']} initial features) were audited to drop {audit['dropped_rows']} incomplete rows, "
    f"standardize {audit['names_fixed']} gene identifiers, and remove {audit['dupes_removed']} duplicate entries. "
    f"For {selected_pair}, differential expression was computed via Welch's t-test and corrected for multiple comparisons using {adj_choice}. "
    f"Features with |log2 fold change| >= {fc_cutoff:.1f} and adjusted p < {padj_cutoff:.3f} "
    f"({int((res_df['Status'] == 'Upregulated').sum())} upregulated, {int((res_df['Status'] == 'Downregulated').sum())} downregulated) "
    f"were visualized via Volcano plot, Ward-clustered Z-score heatmaps, and Venn overlap diagrams exported at {export_dpi} DPI via BioFigureStudio v3."
)