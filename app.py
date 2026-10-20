import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set page config (optional, makes it look clean)
st.set_page_config(page_title="BioFigureStudio", layout="wide")
st.title("BioFigureStudio | Computational Suite")

# 1. Create the tabs (we now have 3 tabs)
tab1, tab2, tab3 = st.tabs([
    "🌋 Volcano Plot Engine", 
    "🧬 PCA Sample Clustering", 
    "🧹 Data Cleaner & Merger"
])

# ==========================================
# TAB 1: Volcano Plot
# ==========================================
with tab1:
    st.header("Volcano Plot Generator")
    st.markdown("Upload your differential expression data to generate a volcano plot.")
    
    # Put your existing volcano plot file uploader and plotting code right here!
    volcano_file = st.file_uploader("Upload Volcano Data CSV", type=["csv"], key="volcano_uploader")
    if volcano_file is not None:
        df_volcano = pd.read_csv(volcano_file)
        st.write("Preview:", df_volcano.head())
        # (Rest of your volcano plot logic...)

# ==========================================
# TAB 2: PCA Cluster Module
# ==========================================
with tab2:
    st.header("Principal Component Analysis (PCA)")
    st.markdown("Upload an expression matrix to check sample clustering.")
    
    pca_file = st.file_uploader("Upload Expression Matrix CSV for PCA", type=["csv"], key="pca_uploader")
    if pca_file is not None:
        pca_df = pd.read_csv(pca_file)
        st.write("Data Preview:", pca_df.head())
        
        numeric_cols = pca_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            st.error("Your dataset needs at least 2 numeric sample columns to perform PCA.")
        else:
            if st.button("Run PCA Analysis"):
                if 'GeneSymbol' in pca_df.columns:
                    expression_data = pca_df.set_index('GeneSymbol')[numeric_cols]
                else:
                    expression_data = pca_df[numeric_cols]
                
                expression_data = expression_data.dropna()
                scaled_data = StandardScaler().fit_transform(expression_data.T)
                
                pca = PCA(n_components=2)
                principal_components = pca.fit_transform(scaled_data)
                
                result_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
                result_df['Sample'] = numeric_cols
                var_exp = pca.explained_variance_ratio_ * 100
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(data=result_df, x='PC1', y='PC2', hue='Sample', s=100, palette='tab10', ax=ax)
                ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% variance)")
                ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% variance)")
                ax.set_title("Sample Clustering PCA")
                ax.grid(True, linestyle='--', alpha=0.5)
                
                st.pyplot(fig)
                st.success("PCA Analysis complete!")

# ==========================================
# TAB 3: Data Cleaner & Merger Module
# ==========================================
with tab3:
    st.header("Automated Spreadsheet Cleaner & Merger")
    st.markdown("Upload two messy CSV files to strip whitespace errors and merge them.")
    
    col1, col2 = st.columns(2)
    with col1:
        file1 = st.file_uploader("Upload Dataset 1", type=["csv"], key="clean_file1")
    with col2:
        file2 = st.file_uploader("Upload Dataset 2", type=["csv"], key="clean_file2")
        
    if file1 is not None and file2 is not None:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        
        st.write("**Dataset 1 Preview:**", df1.head(2))
        st.write("**Dataset 2 Preview:**", df2.head(2))
        
        common_cols = list(set(df1.columns).intersection(set(df2.columns)))
        
        if not common_cols:
            st.error("No matching column names found between the two files!")
        else:
            join_col = st.selectbox("Select Common Column to Merge On:", common_cols)
            
            if st.button("Clean and Merge Datasets"):
                df1.columns = df1.columns.str.strip()
                df2.columns = df2.columns.str.strip()
                
                for df in [df1, df2]:
                    for col in df.select_dtypes(include=['object']).columns:
                        df[col] = df[col].str.strip()
                        
                df1 = df1.dropna(how='all')
                df2 = df2.dropna(how='all')
                
                merged_df = pd.merge(df1, df2, on=join_col, how='inner')
                merged_df = merged_df.fillna(0)
                
                st.success(f"Successfully merged! Final shape: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns.")
                st.dataframe(merged_df.head())
                
                csv_data = merged_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Cleaned & Merged CSV",
                    data=csv_data,
                    file_name="cleaned_merged_output.csv",
                    mime="text/csv"
                )