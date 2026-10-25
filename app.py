import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Farmer Income Prediction", page_icon="🌾", layout="wide")

st.markdown("""
<style>
.main {background: #f7faf7;}
[data-testid="stMetric"] {background: white; border: 1px solid #dbe7dc; padding: 14px; border-radius: 12px;}
.hero {padding: 22px 24px; border-radius: 16px; background: linear-gradient(120deg,#174c35,#3f875c); color:white; margin-bottom:18px;}
.hero h1 {color:white; margin-bottom:4px;}
.hero p {color:#e7f4e9; margin:0;}
</style>
""", unsafe_allow_html=True)

TARGET_CANDIDATES = [
    "Target_Variable/Total Income", "Target_Variable_Total Income",
    "Total Income", "Total_Income", "total_income", "TARGET"
]
DROP_HINTS = {
    "FarmerID", "CITY", "Zipcode", "DISTRICT", "VILLAGE",
    "K022-Nearest Mandi Name", "Location", "Address type", "Ownership"
}

@st.cache_data(show_spinner=False)
def read_csv_file(file_bytes, name):
    import io
    return pd.read_csv(io.BytesIO(file_bytes))

def locate_target(df):
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            return c
    # Be conservative: look for a unique column that clearly names total income.
    matches = [c for c in df.columns if "total income" in str(c).lower() and ("target" in str(c).lower() or "variable" in str(c).lower())]
    return matches[0] if len(matches) == 1 else None

def clean_target(series):
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False).str.replace("₹", "", regex=False), errors="coerce")

def prepare_features(df, target):
    data = df.copy()
    data[target] = clean_target(data[target])
    data = data.dropna(subset=[target])
    drop_cols = []
    for c in data.columns:
        cl = str(c).lower()
        if c != target and (c in DROP_HINTS or cl.endswith("id") or cl in {"farmerid","zipcode"}):
            drop_cols.append(c)
    X = data.drop(columns=[target] + drop_cols, errors="ignore")
    # Remove entirely empty and constant columns.
    X = X.loc[:, X.notna().any()]
    constant_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    X = X.drop(columns=constant_cols, errors="ignore")
    y = data.loc[X.index, target]
    # Avoid extremely high-cardinality free-text columns that make a poor general-purpose demo.
    text_cols = [c for c in X.select_dtypes(include=["object"]).columns if X[c].nunique(dropna=True) > 500]
    X = X.drop(columns=text_cols, errors="ignore")
    return X, y, drop_cols + constant_cols + text_cols

def make_pipeline(model, X):
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2))
    ])
    pre = ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols)
    ], remainder="drop")
    return Pipeline([("preprocess", pre), ("model", model)])

def rupees(v):
    return f"₹{v:,.0f}"

st.markdown("""
<div class="hero">
  <h1>🌾 Farmer Income Prediction Dashboard</h1>
  <p>Explore farmer-income data, compare regression models, and estimate annual income from a prepared dataset.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Data source")
    st.write("Upload the same Kaggle dataset used by the project, or load it from a local CSV.")
    train_file = st.file_uploader("Upload lte_train.csv", type=["csv"], key="train")
    if train_file is not None:
        raw_df = read_csv_file(train_file.getvalue(), train_file.name)
        st.success(f"Loaded {train_file.name}")
    else:
        raw_df = None
    st.caption("Expected training file: lte_train.csv (47,970 rows × 105 columns, per the project README).")
    st.divider()
    st.header("Model settings")
    test_size = st.slider("Test split (%)", min_value=10, max_value=40, value=20, step=5)
    random_state = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
    trees = st.slider("Random Forest trees", min_value=50, max_value=300, value=100, step=50)

if raw_df is None:
    st.info("Upload `lte_train.csv` in the left sidebar to activate the dashboard. The app does not invent a dataset or silently substitute different records.")
    st.markdown("### What you can explore")
    c1,c2,c3 = st.columns(3)
    c1.metric("Project dataset", "47,970 rows", "as documented")
    c2.metric("Target", "Total annual income", "INR")
    c3.metric("Algorithms", "3", "Linear Regression, Decision Tree, Random Forest")
    st.markdown("""
    **Where to get the dataset:** [Farmer Income Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/manishpatil115/farmer-income-prediction-dataset)

    Once downloaded, upload `lte_train.csv`. If your target column has a different name, check the exact column name in the notebook and update the target mapping in this app.
    """)
    st.stop()

target = locate_target(raw_df)
if target is None:
    st.error("Could not identify the target column automatically. Expected `Target_Variable/Total Income` or a clearly named total-income target.")
    st.write("Columns found in uploaded file:")
    st.code(", ".join(map(str, raw_df.columns)))
    target = st.selectbox("Choose the target column manually", options=list(raw_df.columns))
else:
    st.caption(f"Target variable detected: `{target}`")

X, y, removed_cols = prepare_features(raw_df, target)
if len(X) < 30 or X.shape[1] == 0:
    st.error("Not enough usable data/features after cleaning. Check that you uploaded the training CSV and selected the correct target.")
    st.stop()

# Tabs
overview, explore, models, predict, guide = st.tabs(["📊 Overview", "🔎 Explore Data", "🤖 Model Lab", "🧮 Predict Income", "ℹ️ Project Guide"])

with overview:
    st.subheader("Dataset overview")
    a,b,c,d = st.columns(4)
    a.metric("Usable records", f"{len(X):,}")
    b.metric("Input features", f"{X.shape[1]:,}")
    c.metric("Median income", rupees(float(y.median())))
    d.metric("Mean income", rupees(float(y.mean())))
    st.write("Target summary")
    st.dataframe(y.describe().rename("Annual income (INR)").to_frame(), use_container_width=True)
    left,right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(7,4))
        sns.histplot(y, bins=50, ax=ax, color="#2f855a")
        ax.set_title("Distribution of total annual income")
        ax.set_xlabel("Income (INR)")
        st.pyplot(fig); plt.close(fig)
    with right:
        fig, ax = plt.subplots(figsize=(7,4))
        sns.histplot(np.log1p(y.clip(lower=0)), bins=50, ax=ax, color="#d97706")
        ax.set_title("Log1p distribution of income")
        ax.set_xlabel("log(1 + income)")
        st.pyplot(fig); plt.close(fig)
    st.caption("All metrics and plots are calculated from the CSV uploaded in this session.")

with explore:
    st.subheader("Interactive exploratory analysis")
    numeric_df = raw_df.select_dtypes(include=np.number).copy()
    numeric_df[target] = y.reindex(numeric_df.index)
    num_cols = numeric_df.columns.tolist()
    if num_cols:
        selected_num = st.selectbox("Choose a numeric feature", options=[c for c in num_cols if c != target] or num_cols)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.scatter(raw_df.loc[X.index, selected_num], y, alpha=.25, s=12, color="#26734d")
        ax.set_xlabel(selected_num); ax.set_ylabel("Total income (INR)")
        ax.set_title(f"{selected_num} vs total income")
        st.pyplot(fig); plt.close(fig)
    region_candidates = [c for c in X.columns if any(k in c.lower() for k in ["region","state","zone"])]
    if region_candidates:
        region_col = st.selectbox("Group income by", region_candidates)
        grouped = pd.DataFrame({"group":X[region_col].fillna("Unknown").astype(str), "income":y}).groupby("group")["income"].agg(["mean","median","count"]).sort_values("mean", ascending=False).head(20)
        st.dataframe(grouped.style.format({"mean":"₹{:,.0f}","median":"₹{:,.0f}","count":"{:,.0f}"}), use_container_width=True)
        fig, ax = plt.subplots(figsize=(9,4))
        grouped.sort_values("mean")["mean"].plot(kind="barh", ax=ax, color="#438b65")
        ax.set_title(f"Average income by {region_col} (top 20)")
        ax.set_xlabel("Average income (INR)")
        st.pyplot(fig); plt.close(fig)
    if len(numeric_df.columns) > 1:
        st.markdown("**Numeric feature correlation**")
        corr_cols = numeric_df.corr(numeric_only=True).columns.tolist()
        chosen_corr = st.multiselect("Choose columns for heatmap (up to 12)", corr_cols, default=corr_cols[:min(8,len(corr_cols))])
        if chosen_corr:
            fig, ax = plt.subplots(figsize=(9,6))
            sns.heatmap(numeric_df[chosen_corr].corr(), cmap="coolwarm", center=0, annot=len(chosen_corr)<=8, fmt=".2f", ax=ax)
            st.pyplot(fig); plt.close(fig)

with models:
    st.subheader("Train and compare models")
    st.warning("Training runs on the uploaded dataset. For large datasets, model fitting may take a few minutes.")
    if st.button("Train all 3 models", type="primary"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=int(random_state))
        estimators = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=int(random_state), min_samples_leaf=2),
            "Random Forest": RandomForestRegressor(n_estimators=int(trees), random_state=int(random_state), n_jobs=-1, min_samples_leaf=2)
        }
        results, fitted = [], {}
        progress = st.progress(0)
        for i,(name,est) in enumerate(estimators.items()):
            pipe = make_pipeline(est, X_train)
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            results.append({"Model":name, "MAE (INR)":mean_absolute_error(y_test,pred), "RMSE (INR)":mean_squared_error(y_test,pred)**0.5, "R²":r2_score(y_test,pred)})
            fitted[name] = pipe
            progress.progress((i+1)/len(estimators))
        results_df = pd.DataFrame(results).sort_values("R²", ascending=False)
        st.session_state["model_results"] = results_df
        st.session_state["fitted_models"] = fitted
        st.session_state["X_columns"] = X.columns.tolist()
        st.session_state["best_model"] = results_df.iloc[0]["Model"]
        st.session_state["test_data"] = (X_test, y_test)
    if "model_results" in st.session_state:
        results_df = st.session_state["model_results"]
        best = st.session_state["best_model"]
        st.success(f"Best model in this run: {best}")
        st.dataframe(results_df.style.format({"MAE (INR)":"₹{:,.2f}","RMSE (INR)":"₹{:,.2f}","R²":"{:.4f}"}), use_container_width=True)
        fig, ax = plt.subplots(figsize=(8,4))
        sns.barplot(data=results_df, x="Model", y="R²", ax=ax, color="#39865d")
        ax.set_ylim(min(0, float(results_df["R²"].min())-.05), min(1.05, max(.1,float(results_df["R²"].max())+.05)))
        ax.set_title("R² score comparison")
        st.pyplot(fig); plt.close(fig)
        st.download_button("Download metrics as CSV", results_df.to_csv(index=False).encode("utf-8"), "model_metrics.csv", "text/csv")
        st.caption("Results can differ from the notebook if preprocessing, data versions, or model settings differ.")

with predict:
    st.subheader("Estimate annual income")
    if "fitted_models" not in st.session_state:
        st.warning("First open Model Lab and click **Train all 3 models**. The prediction form uses the model trained in this session.")
    else:
        fitted = st.session_state["fitted_models"]
        default_model = st.session_state["best_model"]
        model_name = st.selectbox("Choose trained model", list(fitted.keys()), index=list(fitted.keys()).index(default_model))
        st.write("Enter the feature values. Missing fields are left to the pipeline's imputer.")
        user_row = {}
        cols = st.columns(2)
        for i,col in enumerate(X.columns):
            with cols[i%2]:
                s = X[col]
                if pd.api.types.is_numeric_dtype(s):
                    med = pd.to_numeric(s, errors="coerce").median()
                    med = float(med) if pd.notna(med) else 0.0
                    lo = float(pd.to_numeric(s, errors="coerce").min()) if pd.notna(pd.to_numeric(s, errors="coerce").min()) else med
                    hi = float(pd.to_numeric(s, errors="coerce").max()) if pd.notna(pd.to_numeric(s, errors="coerce").max()) else med
                    if lo == hi: lo, hi = med-1, med+1
                    user_row[col] = st.number_input(str(col), value=med, min_value=lo, max_value=hi, key=f"pred_{col}")
                else:
                    vals = s.dropna().astype(str).value_counts().head(100).index.tolist()
                    vals = vals or ["Unknown"]
                    user_row[col] = st.selectbox(str(col), vals, key=f"pred_{col}")
        if st.button("Predict annual income", type="primary"):
            pred = float(fitted[model_name].predict(pd.DataFrame([user_row], columns=X.columns))[0])
            st.metric("Predicted total annual income", rupees(pred))
            st.caption("This is an educational model estimate, not an official income certificate or a guarantee of actual income.")

with guide:
    st.subheader("Project details")
    st.markdown("""
    **Project:** Farmer Income Prediction  
    **Programme:** IBM SkillsBuild Data Analytics with AI Internship  
    **Internship period:** 17 August 2026 – 30 September 2026  
    **Dataset source:** [Kaggle Farmer Income Prediction Dataset](https://www.kaggle.com/datasets/manishpatil115/farmer-income-prediction-dataset)  
    **Tech stack:** Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Streamlit  
    **Algorithms:** Linear Regression, Decision Tree Regressor, Random Forest Regressor  

    The original project report recorded the Random Forest Regressor as the best of the three tested models, with R² = 0.9381, MAE = ₹260,292.32 and RMSE = ₹522,759.80. This live dashboard retrains models from the uploaded data, so its results may differ.
    """)
    st.markdown("**Important:** the app does not include or publish farmer-level data. Upload the dataset privately when using the dashboard.")
