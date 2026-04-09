import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import numpy as np

from sklearn.preprocessing import TargetEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.decomposition import PCA

from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler
from imblearn.under_sampling import TomekLinks, RandomUnderSampler, NearMiss

st.set_page_config(page_title="ML Dashboard", layout="wide")

BALANCE_METHODS = {
    "SMOTEENN (Over)":           lambda: SMOTEENN(random_state=42),
    "BorderlineSMOTE (Over)":    lambda: BorderlineSMOTE(random_state=42),
    "RandomOverSampler (Over)":  lambda: RandomOverSampler(random_state=42),
    "TomekLinks (Under)":        lambda: TomekLinks(),
    "RandomUnderSampler (Under)": lambda: RandomUnderSampler(random_state=42),
    "NearMiss (Under)":          lambda: NearMiss(),
    "SMOTETomek (Balance)":      lambda: SMOTETomek(random_state=42),
    "SMOTEENN (Balance)":        lambda: SMOTEENN(random_state=42),
    "Без балансировки":          None,
}

GROUPS = {
    "Over-sampling": ["SMOTEENN (Over)", "BorderlineSMOTE (Over)", "RandomOverSampler (Over)"],
    "Under-sampling": ["TomekLinks (Under)", "RandomUnderSampler (Under)", "NearMiss (Under)"],
    "Balance": ["SMOTETomek (Balance)", "SMOTEENN (Balance)"],
}

@st.cache_data
def load_default():
    return pd.read_csv("data/bank-full.csv", sep=";")

def load_uploaded(file):
    try:
        df = pd.read_csv(file, sep=None, engine="python")
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")
        return None

def process_df(df, target_col):
    df = df.copy()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)

    ohe_cols = [c for c in cat_cols if c != "job" and df[c].nunique() <= 15]
    if ohe_cols:
        df = pd.get_dummies(df, columns=ohe_cols, dtype=int)

    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col].astype(str))

    if "job" in df.columns:
        te = TargetEncoder()
        df[["job"]] = te.fit_transform(df[["job"]], df[target_col])

    if "month" in df.columns:
        month_map = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
                     'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
        df["month"] = df["month"].map(month_map).fillna(df["month"])
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df = df.drop("month", axis=1)

    if "day" in df.columns:
        df["day_sin"] = np.sin(2 * np.pi * df["day"] / 12)
        df["day_cos"] = np.cos(2 * np.pi * df["day"] / 12)
        df = df.drop("day", axis=1)

    leftover = df.select_dtypes(include="object").columns.tolist()
    if leftover:
        df = df.drop(columns=leftover)

    return df

def apply_balance(x_train, y_train, method_name):
    factory = BALANCE_METHODS.get(method_name)
    if factory is None:
        return x_train, y_train
    return factory().fit_resample(x_train, y_train)

def get_splits(df, target_col):
    x = df.drop(target_col, axis=1)
    y = df[target_col]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    scaled_cols = x_train.select_dtypes(include=[np.number]).columns.tolist()
    ssc = StandardScaler()
    x_train = x_train.copy()
    x_test = x_test.copy()
    x_train[scaled_cols] = ssc.fit_transform(x_train[scaled_cols])
    x_test[scaled_cols] = ssc.transform(x_test[scaled_cols])
    return x_train, x_test, y_train, y_test

def train_model(x_train, y_train, x_test, params):
    rf = RandomForestClassifier(**params, random_state=42, n_jobs=-1, class_weight="balanced")
    rf.fit(x_train, y_train)
    return rf.predict(x_test)

def get_metrics(y_test, preds, label=""):
    return {
        "Метод": label,
        "Accuracy": round(accuracy_score(y_test, preds), 4),
        "F1": round(f1_score(y_test, preds, average="weighted", zero_division=0), 4),
        "Precision": round(precision_score(y_test, preds, average="weighted", zero_division=0), 4),
        "Recall": round(recall_score(y_test, preds, average="weighted", zero_division=0), 4),
    }

def show_method_viz(method_name, x_tr, y_tr):
    st.subheader(f"Метод: **{method_name}**")
    try:
        x_res, y_res = apply_balance(x_tr, y_tr, method_name)

        col1, col2 = st.columns(2)

        # Распределение классов
        with col1:
            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
            colors = ["#4C72B0", "#DD8452"]
            vc_b = pd.Series(y_tr).value_counts().sort_index()
            vc_a = pd.Series(y_res).value_counts().sort_index()
            axes[0].bar([str(i) for i in vc_b.index], vc_b.values,
                        color=colors[:len(vc_b)], edgecolor="white")
            axes[0].set_title("До")
            axes[1].bar([str(i) for i in vc_a.index], vc_a.values,
                        color=colors[:len(vc_a)], edgecolor="white")
            axes[1].set_title("После")
            plt.suptitle("Распределение классов", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # PCA scatter
        with col2:
            pca = PCA(n_components=2, random_state=42)
            x_b_pca = pca.fit_transform(x_tr.values)
            x_a_pca = pca.transform(x_res)
            y_tr_arr = np.array(y_tr)
            y_res_arr = np.array(y_res)

            fig2, axes2 = plt.subplots(1, 2, figsize=(8, 3))
            for cls in np.unique(y_tr_arr):
                m = y_tr_arr == cls
                axes2[0].scatter(x_b_pca[m, 0], x_b_pca[m, 1],
                                 label=f"Кл.{cls}", alpha=0.3, s=5, color=colors[cls % 2])
            axes2[0].set_title("PCA До")
            axes2[0].legend(markerscale=3, fontsize=7)

            for cls in np.unique(y_res_arr):
                m = y_res_arr == cls
                axes2[1].scatter(x_a_pca[m, 0], x_a_pca[m, 1],
                                 label=f"Кл.{cls}", alpha=0.3, s=5, color=colors[cls % 2])
            axes2[1].set_title("PCA После")
            axes2[1].legend(markerscale=3, fontsize=7)

            plt.suptitle("Пространство признаков (PCA)", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        before = pd.Series(y_tr).value_counts().to_dict()
        after = pd.Series(y_res).value_counts().to_dict()
        st.caption(f"Размер до: {dict(before)} → после: {dict(after)}")

    except Exception as e:
        st.error(f"Ошибка для {method_name}: {e}")

st.sidebar.title("Настройки")

st.sidebar.markdown("Данные")
uploaded_file = st.sidebar.file_uploader("Загрузить CSV", type=["csv"])

if uploaded_file:
    df_raw = load_uploaded(uploaded_file)
    if df_raw is None:
        st.stop()
    st.sidebar.success(f"Загружен: {uploaded_file.name}")
else:
    try:
        df_raw = load_default()
        st.sidebar.info("Используется bank-full.csv")
    except:
        st.sidebar.warning("Загрузите CSV файл")
        st.stop()

target_col = st.sidebar.selectbox("Целевая колонка", df_raw.columns.tolist(),
                                   index=len(df_raw.columns)-1)

st.sidebar.markdown("###Балансировка")
balance_method = st.sidebar.selectbox("Метод", list(BALANCE_METHODS.keys()))

page = st.sidebar.radio("Страница", ["Dataset", "Visual Analysis", "Sampling Visualization", "Model"])

if page == "Dataset":
    st.title("Датасет")
    st.dataframe(df_raw.head(20), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Строк", df_raw.shape[0])
    c2.metric("Колонок", df_raw.shape[1])
    c3.metric("Пропусков", int(df_raw.isnull().sum().sum()))
    st.subheader("Статистика")
    st.write(df_raw.describe())
    st.subheader("Типы данных")
    st.write(pd.DataFrame({"Тип": df_raw.dtypes, "Уникальных": df_raw.nunique()}))


elif page == "Visual Analysis":
    st.title("Визуальный анализ данных")
    df_proc = process_df(df_raw, target_col)

    st.header("1. Распределение целевой переменной")
    fig, ax = plt.subplots(figsize=(6, 4))
    vc = df_raw[target_col].astype(str).value_counts()
    ax.bar(vc.index, vc.values, color=["#4C72B0", "#DD8452"], edgecolor="white")
    ax.set_ylabel("Количество")
    st.pyplot(fig)
    plt.close()

    st.header("2. Корреляционная матрица")
    num_cols = df_proc.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) > 1:
        corr = df_proc[num_cols].corr()
        fig2, ax2 = plt.subplots(figsize=(min(len(num_cols), 16), min(len(num_cols), 12)))
        sns.heatmap(corr, annot=len(num_cols) <= 15, fmt=".2f",
                    cmap="coolwarm", ax=ax2, linewidths=0.5)
        st.pyplot(fig2)
        plt.close()

    st.header("3. Гистограммы")
    raw_num = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    if raw_num:
        sel = st.multiselect("Колонки", raw_num, default=raw_num[:min(4, len(raw_num))])
        if sel:
            cols_r = 3
            rows_r = (len(sel) + cols_r - 1) // cols_r
            fig3, axes3 = plt.subplots(rows_r, cols_r, figsize=(15, 4 * rows_r))
            axes3 = np.array(axes3).flatten()
            for i, col in enumerate(sel):
                axes3[i].hist(df_raw[col].dropna(), bins=30, color="#4C72B0", edgecolor="white")
                axes3[i].set_title(col)
            for j in range(i+1, len(axes3)):
                axes3[j].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

    st.header("4. Boxplot - выбросы")
    if raw_num:
        sel_b = st.multiselect("Колонки", raw_num, default=raw_num[:min(4, len(raw_num))], key="box")
        if sel_b:
            fig4, axes4 = plt.subplots(1, len(sel_b), figsize=(4 * len(sel_b), 5))
            if len(sel_b) == 1:
                axes4 = [axes4]
            for i, col in enumerate(sel_b):
                axes4[i].boxplot(df_raw[col].dropna(), patch_artist=True,
                                 boxprops=dict(facecolor="#4C72B0"),
                                 medianprops=dict(color="#DD8452", linewidth=2))
                axes4[i].set_title(col)
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close()

elif page == "Sampling Visualization":
    st.title("Визуализация методов балансировки")
    st.markdown("Для каждого метода показываем **распределение классов** и **PCA scatter** до и после.")

    df_proc = process_df(df_raw, target_col)
    x_train, x_test, y_train, y_test = get_splits(df_proc, target_col)

    tab_over, tab_under, tab_bal = st.tabs(["Over-sampling", "Under-sampling", "Balance"])

    with tab_over:
        for m in GROUPS["Over-sampling"]:
            show_method_viz(m, x_train, y_train)
            st.divider()

    with tab_under:
        for m in GROUPS["Under-sampling"]:
            show_method_viz(m, x_train, y_train)
            st.divider()

    with tab_bal:
        for m in GROUPS["Balance"]:
            show_method_viz(m, x_train, y_train)
            st.divider()

elif page == "Model":
    st.title("Обучение модели")
    df_proc = process_df(df_raw, target_col)

    st.header("Параметры модели")
    c1, c2 = st.columns(2)
    with c1:
        n_estimators = st.slider("Деревьев", 10, 200, 100, 10)
        max_depth = st.slider("Макс. глубина", 2, 30, 10)
        random_state = st.number_input("Random state", 0, 100, 42)
    with c2:
        min_samples_split = st.slider("Min samples split", 2, 10, 2)
        min_samples_leaf = st.slider("Min samples leaf", 1, 10, 1)

    params = dict(n_estimators=n_estimators, max_depth=max_depth,
                  min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)

    st.info(f"Метод балансировки: **{balance_method}**")
    x_train, x_test, y_train, y_test = get_splits(df_proc, target_col)

    if st.button("▶ Обучить модель"):
        with st.spinner("Обучаю..."):
            try:
                x_tr_b, y_tr_b = apply_balance(x_train, y_train, balance_method)
                preds = train_model(x_tr_b, y_tr_b, x_test, params)
                st.success("Готово!")
                st.dataframe(pd.DataFrame([get_metrics(y_test, preds, balance_method)]),
                             use_container_width=True)
            except Exception as e:
                st.error(f"Ошибка: {e}")

    st.divider()
    st.header("Сравнить все методы балансировки")

    if st.button("Сравнить все методы"):
        all_metrics = []
        progress = st.progress(0)
        methods = list(BALANCE_METHODS.keys())
        for i, method in enumerate(methods):
            with st.spinner(f"{method}..."):
                try:
                    x_tr_b, y_tr_b = apply_balance(x_train, y_train, method)
                    preds = train_model(x_tr_b, y_tr_b, x_test, params)
                    all_metrics.append(get_metrics(y_test, preds, method))
                except Exception as e:
                    all_metrics.append({"Метод": method, "Accuracy": None,
                                        "F1": None, "Precision": None, "Recall": None})
            progress.progress((i + 1) / len(methods))

        results_df = pd.DataFrame(all_metrics)
        st.subheader("Таблица метрик")
        st.dataframe(results_df, use_container_width=True)

        plot_df = results_df.dropna(subset=["F1"])
        if not plot_df.empty:
            metric_cols = ["Accuracy", "F1", "Precision", "Recall"]
            colors = plt.cm.Set2(np.linspace(0, 1, len(plot_df)))
            fig, axes = plt.subplots(1, 4, figsize=(18, 5))
            for i, metric in enumerate(metric_cols):
                axes[i].bar(range(len(plot_df)), plot_df[metric], color=colors, edgecolor="white")
                axes[i].set_title(metric, fontsize=13)
                axes[i].set_xticks(range(len(plot_df)))
                axes[i].set_xticklabels(plot_df["Метод"], rotation=40, ha="right", fontsize=8)
                axes[i].set_ylim(0, 1.1)
                for j, v in enumerate(plot_df[metric]):
                    if v is not None:
                        axes[i].text(j, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
            plt.suptitle("Сравнение всех методов балансировки", fontsize=14, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            best_f1 = plot_df.loc[plot_df["F1"].idxmax(), "Метод"]
            best_acc = plot_df.loc[plot_df["Accuracy"].idxmax(), "Метод"]
            st.success(f"Лучший по F1: **{best_f1}** | Лучший по Accuracy: **{best_acc}**")
