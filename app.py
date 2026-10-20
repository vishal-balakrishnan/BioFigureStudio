import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
import io

# Page Config
st.set_page_config(page_title="BioFigureStudio | Volcano Plot Engine", layout="centered")

st.title("🧬 BioFigureStudio: Publication-Ready Volcano Generator")
st.markdown("Upload your differential expression results CSV/TSV, adjust your thresholds, and export 300-DPI vector figures instantly.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("1. Data Mapping")
gene_col = st.sidebar.text_input("Gene ID Column", value="GeneSymbol")
fc_col = st.sidebar.text_input("log2 Fold Change Column", value="log2FoldChange")
pval_col = st.sidebar.text_input("p-value Column", value="padj")

st.sidebar.header("2. Filtering Thresholds")
fc_thresh = st.sidebar.slider("Absolute log2FC Threshold", 0.5, 3.0, 1.5, 0.1)
p_thresh = st.sidebar.slider("Adjusted p-value Cutoff", 0.001, 0.1, 0.05, 0.005)
top_n = st.sidebar.slider("Top Genes to Label", 5, 40, 15, 1)

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload your expression results (CSV or TSV)", type=["csv", "tsv", "txt"])

if uploaded_file is not None:
    # Read data
    sep = "\t" if uploaded_file.name.endswith((".tsv", ".txt")) else ","
    try:
        df = pd.read_csv(uploaded_file, sep=sep)
        st.success(f"Successfully loaded file: {uploaded_file.name} ({len(df)} rows)")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # Validate columns
    required_cols = [gene_col, fc_col, pval_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns in CSV: {missing_cols}. Update sidebar mapping.")
        st.stop()

    # --- SANITIZE DATA (Same engine logic) ---
    initial_rows = len(df)
    df_clean = df.dropna(subset=required_cols).copy()
    df_clean = df_clean.sort_values(by=pval_col, ascending=True).drop_duplicates(subset=[gene_col], keep="first")
    
    min_nonzero_p = df_clean.loc[df_clean[pval_col] > 0, pval_col].min()
    df_clean[pval_col] = df_clean[pval_col].clip(lower=min_nonzero_p * 0.1)
    df_clean["neg_log10_p"] = -np.log10(df_clean[pval_col])
    
    removed_rows = initial_rows - len(df_clean)
    st.info(f"Data Sanitization: Kept {len(df_clean)} valid genes (Removed {removed_rows} rows with NaNs/duplicates).")

    # Classify genes
    conditions = [
        (df_clean[fc_col] >= fc_thresh) & (df_clean[pval_col] <= p_thresh),
        (df_clean[fc_col] <= -fc_thresh) & (df_clean[pval_col] <= p_thresh)
    ]
    choices = ["Upregulated", "Downregulated"]
    df_clean["Status"] = np.select(conditions, choices, default="Not Significant")

    # --- RENDER PLOT ---
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    palette = {
        "Upregulated": "#E64B35",
        "Downregulated": "#4DBBD5",
        "Not Significant": "#D3D3D3"
    }

    for status in ["Not Significant", "Downregulated", "Upregulated"]:
        sub = df_clean[df_clean["Status"] == status]
        alpha = 0.45 if status == "Not Significant" else 0.85
        size = 18 if status == "Not Significant" else 36
        ax.scatter(
            sub[fc_col], sub["neg_log10_p"],
            c=palette[status], label=f"{status} ({len(sub)})",
            s=size, alpha=alpha, edgecolors="none"
        )

    ax.axhline(-np.log10(p_thresh), color="#404040", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.axvline(fc_thresh, color="#404040", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.axvline(-fc_thresh, color="#404040", linestyle="--", linewidth=0.9, alpha=0.7)

    # Label top genes
    sig_df = df_clean[df_clean["Status"] != "Not Significant"].sort_values(by="neg_log10_p", ascending=False).head(top_n)
    texts = []
    for _, row in sig_df.iterrows():
        texts.append(
            ax.text(
                row[fc_col], row["neg_log10_p"], str(row[gene_col]),
                fontsize=8.5, fontweight="semibold", color="#1A1A1A"
            )
        )

    if texts:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#555555", lw=0.6, alpha=0.8), expand_points=(1.4, 1.4))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(r"$\log_2(\mathrm{Fold\ Change})$", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"$-\log_{10}(\mathrm{Adjusted\ } P\mathrm{-value})$", fontsize=11, fontweight="bold")
    ax.set_title("Differential Expression Profile", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True, facecolor="white", edgecolor="#E0E0E0", fontsize=9, loc="upper left")
    plt.tight_layout()

    # Display in web app
    st.pyplot(fig)

    # --- DOWNLOAD BUTTONS ---
    buf_png = io.BytesIO()
    fig.savefig(buf_png, format="png", dpi=300)
    buf_png.seek(0)

    buf_pdf = io.BytesIO()
    fig.savefig(buf_pdf, format="pdf")
    buf_pdf.seek(0)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Download 300-DPI PNG", data=buf_png, file_name="volcano_plot_300dpi.png", mime="image/png")
    with col2:
        st.download_button("📥 Download Vector PDF", data=buf_pdf, file_name="volcano_plot.pdf", mime="application/pdf")

else:
    st.info("👆 Upload your `dirty_deseq2_results.csv` file using the sidebar or drag-and-drop above to test the app.")
    import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
import io
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Page Config
st.set_page_config(page_title="BioFigureStudio | Volcano Plot Engine", layout="centered")

st.title("BioFigureStudio: Publication-Ready Volcano Generator")
st.markdown("Upload your differential expression results CSV/TSV, adjust your thresholds, and export 300-DPI vector figures instantly.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("1. Data Mapping")
gene_col = st.sidebar.text_input("Gene ID Column", value="GeneSymbol")
fc_col = st.sidebar.text_input("log2 Fold Change Column", value="log2FoldChange")
pval_col = st.sidebar.text_input("p-value Column", value="padj")

st.sidebar.header("2. Filtering Thresholds")
fc_thresh = st.sidebar.slider("Absolute log2FC Threshold", 0.5, 3.0, 1.5, 0.1)
p_thresh = st.sidebar.slider("Adjusted p-value Cutoff", 0.001, 0.1, 0.05, 0.005)
top_n = st.sidebar.slider("Top Genes to Label", 5, 40, 15, 1)

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload your expression results (CSV or TSV)", type=["csv", "tsv", "txt"])

if uploaded_file is not None:
    # Read data
    sep = "\t" if uploaded_file.name.endswith((".tsv", ".txt")) else ","
    try:
        df = pd.read_csv(uploaded_file, sep=sep)
        st.success(f"Successfully loaded file: {uploaded_file.name} ({len(df)} rows)")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # Validate columns
    required_cols = [gene_col, fc_col, pval_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns in CSV: {missing_cols}. Update sidebar mapping.")
        st.stop()

    # --- SANITIZE DATA (Same engine logic) ---
    initial_rows = len(df)
    df_clean = df.dropna(subset=required_cols).copy()
    df_clean = df_clean.sort_values(by=pval_col, ascending=True).drop_duplicates(subset=[gene_col], keep="first")
    
    min_nonzero_p = df_clean.loc[df_clean[pval_col] > 0, pval_col].min()
    df_clean[pval_col] = df_clean[pval_col].clip(lower=min_nonzero_p * 0.1)
    df_clean["neg_log10_p"] = -np.log10(df_clean[pval_col])
    
    removed_rows = initial_rows - len(df_clean)
    st.info(f"Data Sanitization: Kept {len(df_clean)} valid genes (Removed {removed_rows} rows with NaNs/duplicates).")

    # Classify genes
    conditions = [
        (df_clean[fc_col] >= fc_thresh) & (df_clean[pval_col] <= p_thresh),
        (df_clean[fc_col] <= -fc_thresh) & (df_clean[pval_col] <= p_thresh)
    ]
    choices = ["Upregulated", "Downregulated"]
    df_clean["Status"] = np.select(conditions, choices, default="Not Significant")

    # --- RENDER PLOT ---
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    palette = {
        "Upregulated": "#E64B35",
        "Downregulated": "#4DBBD5",
        "Not Significant": "#D3D3D3"
    }

    for status in ["Not Significant", "Downregulated", "Upregulated"]:
        sub = df_clean[df_clean["Status"] == status]
        alpha = 0.45 if status == "Not Significant" else 0.85
        size = 18 if status == "Not Significant" else 36
        ax.scatter(
            sub[fc_col], sub["neg_log10_p"],
            c=palette[status], label=f"{status} ({len(sub)})",
            s=size, alpha=alpha, edgecolors="none"
        )

    ax.axhline(-np.log10(p_thresh), color="#404040", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.axvline(fc_thresh, color="#404040", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.axvline(-fc_thresh, color="#404040", linestyle="--", linewidth=0.9, alpha=0.7)

    # Label top genes
    sig_df = df_clean[df_clean["Status"] != "Not Significant"].sort_values(by="neg_log10_p", ascending=False).head(top_n)
    texts = []
    for _, row in sig_df.iterrows():
        texts.append(
            ax.text(
                row[fc_col], row["neg_log10_p"], str(row[gene_col]),
                fontsize=8.5, fontweight="semibold", color="#1A1A1A"
            )
        )

    if texts:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#555555", lw=0.6, alpha=0.8), expand_points=(1.4, 1.4))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(r"$\log_2(\mathrm{Fold\ Change})$", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"$-\log_{10}(\mathrm{Adjusted\ } P\mathrm{-value})$", fontsize=11, fontweight="bold")
    ax.set_title("Differential Expression Profile", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=True, facecolor="white", edgecolor="#E0E0E0", fontsize=9, loc="upper left")
    plt.tight_layout()

    # Display in web app
    st.pyplot(fig)

    # --- DOWNLOAD BUTTONS ---
    buf_png = io.BytesIO()
    fig.savefig(buf_png, format="png", dpi=300)
    buf_png.seek(0)

    buf_pdf = io.BytesIO()
    fig.savefig(buf_pdf, format="pdf")
    buf_pdf.seek(0)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Download 300-DPI PNG", data=buf_png, file_name="volcano_plot_300dpi.png", mime="image/png")
    with col2:
        st.download_button("📥 Download Vector PDF", data=buf_pdf, file_name="volcano_plot.pdf", mime="application/pdf")

else:
    st.info(" Upload your `dirty_deseq2_results.csv` file using the sidebar or drag-and-drop above to test the app.")
    st.title("BioFigureStudio | Computational Suite")

# Create tabs for different bioinformatics modules
tab1, tab2 = st.tabs([" Volcano Plot Engine", " PCA Sample Clustering"])

with tab1:
    st.header("Volcano Plot Generator")
    # (Your existing volcano plot code and uploader go right here!)
    # ...

with tab2:
    st.header("Principal Component Analysis (PCA)")
    st.markdown("Upload a raw expression matrix (Genes as rows, Samples/Patients as columns) to check sample variance and clustering.")
    
    pca_file = st.file_uploader("Upload Expression Matrix CSV for PCA", type=["csv"], key="pca_uploader")
    
    if pca_file is not None:
        pca_df = pd.read_csv(pca_file)
        st.write("Data Preview:", pca_df.head())
        
        # Assume first column is Gene IDs/Symbols and rest are numeric sample columns
        numeric_cols = pca_df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            st.error("Your dataset needs at least 2 numeric sample columns to perform PCA.")
        else:
            if st.button("Run PCA Analysis"):
                # Prepare data: set gene names as index if string column exists
                if 'GeneSymbol' in pca_df.columns:
                    expression_data = pca_df.set_index('GeneSymbol')[numeric_cols]
                else:
                    expression_data = pca_df[numeric_cols]
                
                # Drop missing values
                expression_data = expression_data.dropna()
                
                # Transpose so samples are rows and genes are features
                scaled_data = StandardScaler().fit_transform(expression_data.T)
                
                pca = PCA(n_components=2)
                principal_components = pca.fit_transform(scaled_data)
                
                result_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
                result_df['Sample'] = numeric_cols
                
                var_exp = pca.explained_variance_ratio_ * 100
                
                # Plotting PCA
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(
                    data=result_df, x='PC1', y='PC2', hue='Sample', s=100, palette='tab10', ax=ax
                )
                ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% variance)")
                ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% variance)")
                ax.set_title("Sample Clustering PCA")
                ax.grid(True, linestyle='--', alpha=0.5)
                
                st.pyplot(fig)
                st.success("PCA Analysis complete! Your samples are successfully mapped.")
