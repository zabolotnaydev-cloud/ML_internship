import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error

st.set_page_config(page_title="Universal ML App", layout="wide")

st.markdown("""
<style>
.main {background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);}
h1,h2,h3 {color:#00eaff}
.stButton>button {
background: linear-gradient(90deg,#00f5ff,#00ff87);
color:black;border-radius:8px;border:none;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_file(file):
    return pd.read_csv(file, sep=None, engine="python")

st.sidebar.title("Settings")

file = st.sidebar.file_uploader("Upload dataset", type=["csv"])

if file is None:
    st.warning("Upload dataset")
    st.stop()

df = load_file(file)

target = st.sidebar.selectbox("Target column", df.columns)

task = st.sidebar.selectbox("Task", ["Auto", "Classification", "Regression"])

page = st.sidebar.radio("Page", ["Data", "Analysis", "Model", "Predict"])

def preprocess(df, target):
    df = df.copy()

    for col in df.select_dtypes(include="object").columns:
        if col != target:
            if df[col].nunique() < 20:
                df = pd.get_dummies(df, columns=[col])
            else:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

    if df[target].dtype == "object":
        le = LabelEncoder()
        df[target] = le.fit_transform(df[target].astype(str))

    df = df.dropna()
    return df

if page == "Data":
    st.title("Dataset")
    st.dataframe(df.head(100))

    c1,c2,c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing", int(df.isna().sum().sum()))

    st.dataframe(df.describe())

if page == "Analysis":
    st.title("Analysis")

    st.subheader("Target distribution")
    fig, ax = plt.subplots()
    df[target].astype(str).value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)

    num_cols = df.select_dtypes(include=np.number).columns

    if len(num_cols) > 1:
        st.subheader("Correlation")
        fig2, ax2 = plt.subplots(figsize=(8,6))
        sns.heatmap(df[num_cols].corr(), cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)

    st.subheader("Histograms")
    sel = st.multiselect("Columns", num_cols, default=num_cols[:3])
    if sel:
        fig3, ax3 = plt.subplots(len(sel), 1, figsize=(6,4*len(sel)))
        if len(sel)==1:
            ax3=[ax3]
        for i,col in enumerate(sel):
            ax3[i].hist(df[col], bins=30)
            ax3[i].set_title(col)
        st.pyplot(fig3)

if page == "Model":
    st.title("Training")

    df_proc = preprocess(df, target)

    X = df_proc.drop(target, axis=1)
    y = df_proc[target]

    if task == "Auto":
        if y.nunique() < 20:
            task_type = "Classification"
        else:
            task_type = "Regression"
    else:
        task_type = task

    test_size = st.slider("Test size", 0.1, 0.4, 0.2)
    n_estimators = st.slider("Trees", 50, 300, 100)
    max_depth = st.slider("Depth", 2, 30, 10)

    if st.button("Train"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if task_type == "Classification":
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        else:
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.columns = X.columns

        if task_type == "Classification":
            st.metric("Accuracy", round(accuracy_score(y_test, preds),3))
            st.metric("F1", round(f1_score(y_test, preds, average="weighted"),3))
        else:
            st.metric("R2", round(r2_score(y_test, preds),3))
            st.metric("RMSE", round(np.sqrt(mean_squared_error(y_test, preds)),3))

        fig, ax = plt.subplots()
        ax.scatter(y_test, preds, alpha=0.3)
        ax.set_title("Real vs Predicted")
        st.pyplot(fig)

        imp = model.feature_importances_
        fig2, ax2 = plt.subplots()
        ax2.barh(X.columns, imp)
        st.pyplot(fig2)

if page == "Predict":
    st.title("Prediction")

    if "model" not in st.session_state:
        st.warning("Train model first")
    else:
        model = st.session_state.model
        scaler = st.session_state.scaler
        cols = st.session_state.columns

        tab1, tab2 = st.tabs(["Manual", "File"])

        with tab1:
            data = {}
            for c in cols:
                data[c] = st.number_input(c, value=0.0)

            if st.button("Predict"):
                df_pred = pd.DataFrame([data])
                df_pred = scaler.transform(df_pred)
                pred = model.predict(df_pred)[0]
                st.success(pred)

        with tab2:
            f = st.file_uploader("Upload test file", type=["csv"])
            if f:
                test = pd.read_csv(f)
                test = test[cols]
                test_scaled = scaler.transform(test)
                preds = model.predict(test_scaled)
                test["prediction"] = preds
                st.dataframe(test)

                fig3, ax3 = plt.subplots()
                ax3.hist(preds, bins=30)
                st.pyplot(fig3)
