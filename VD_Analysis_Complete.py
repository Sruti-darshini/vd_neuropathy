
"""
Vitamin D and Neuropathy Analysis Pipeline

This script covers:
- Dataset preprocessing
- Statistical tests (T-test, Wilcoxon, ANOVA, Kruskal-Wallis, KS)
- Machine learning model evaluation
- Clustering and SHAP analysis
- Data augmentation (SMOTE, GMM, IV perturbation)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_auc_score,
    f1_score, precision_score, recall_score, brier_score_loss, fbeta_score, roc_curve
)
from sklearn.decomposition import PCA
from scipy.stats import ttest_rel, wilcoxon, f_oneway, kruskal, ks_2samp, mode
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

def load_clean_data(filepath):
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.strip()
    return df.dropna()

def basic_summary(df):
    return df.describe(include="all"), df.isnull().sum(), df.corr()

def perform_pairwise_tests(df1, df2):
    t_results, w_results = {}, {}
    common = list(set(df1.columns) & set(df2.columns))
    df1, df2 = df1[common].dropna(), df2[common].dropna()
    for col in common:
        if df1[col].dtype in ['float64', 'int64']:
            t_stat, t_p = ttest_rel(df1[col], df2[col])
            try:
                w_stat, w_p = wilcoxon(df1[col], df2[col])
            except ValueError:
                w_stat, w_p = None, None
            t_results[col] = {"T-Stat": t_stat, "P-Value": t_p}
            w_results[col] = {"W-Stat": w_stat, "P-Value": w_p}
    return pd.DataFrame(t_results).T, pd.DataFrame(w_results).T

def perform_groupwise_tests(df1, df2, df3):
    anova, kruskal_res = {}, {}
    common = list(set(df1.columns) & set(df2.columns) & set(df3.columns))
    for col in common:
        if all(df[col].dtype in ['float64', 'int64'] for df in [df1, df2, df3]):
            a_stat, a_p = f_oneway(df1[col], df2[col], df3[col])
            k_stat, k_p = kruskal(df1[col], df2[col], df3[col])
            anova[col] = {"F-Stat": a_stat, "P-Value": a_p}
            kruskal_res[col] = {"Kruskal Stat": k_stat, "P-Value": k_p}
    return pd.DataFrame(anova).T, pd.DataFrame(kruskal_res).T

def train_classifiers(X, y):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(kernel='linear', probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(64,32), max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "KNN": KNeighborsClassifier(),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": lgb.LGBMClassifier()
    }
    results = {}
    X_train, X_test, y_train, y_test = train_test_split(
        StandardScaler().fit_transform(X), y, test_size=0.2, stratify=y, random_state=42
    )
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = accuracy_score(y_test, model.predict(X_test))
    return results

def apply_clustering(X, n_clusters=3):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    return kmeans.fit_predict(X_scaled), agglomerative.fit_predict(X_scaled)

def generate_smote_data(X, y, n_samples=500):
    X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X, y)
    df_syn = pd.DataFrame(X_resampled, columns=X.columns)
    df_syn['Target'] = y_resampled
    return df_syn.sample(n=n_samples, random_state=42)

def perturb_data(X, std=0.05):
    return X + np.random.normal(0, std, X.shape)

def augment_data(X, y, n=100):
    X_aug, y_aug = [], []
    for i in range(X.shape[0]):
        for _ in range(n):
            noise = np.random.normal(0, 0.05, X.shape[1])
            X_aug.append(X[i] + noise)
            y_aug.append(y.iloc[i])
    return pd.DataFrame(X_aug, columns=X.columns), pd.Series(y_aug)

def run_ks_test(df, group1, group2):
    result = {}
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            stat, p = ks_2samp(group1[col], group2[col])
            result[col] = {"KS Stat": stat, "p-value": p}
    return pd.DataFrame(result).T

if __name__ == "__main__":
    df1 = load_clean_data("Without PN - Copy.xlsx")
    df2 = load_clean_data("Before VD With PN - Copy.xlsx")
    df3 = load_clean_data("After VD With PN - Copy.xlsx")
    combined = pd.concat([df1, df2, df3], ignore_index=True)
    combined.to_excel("Combined_Dataset.xlsx", index=False)

    t_df, w_df = perform_pairwise_tests(df2, df3)
    t_df.to_excel("T_test.xlsx")
    w_df.to_excel("Wilcoxon.xlsx")

    a_df, k_df = perform_groupwise_tests(df1, df2, df3)
    a_df.to_excel("ANOVA.xlsx")
    k_df.to_excel("Kruskal.xlsx")

    combined["Target"] = np.random.randint(0, 3, size=len(combined))
    X = combined.drop(columns=["Target"], errors='ignore')
    y = combined["Target"]
    acc_results = train_classifiers(X, y)
    pd.DataFrame.from_dict(acc_results, orient="index", columns=["Accuracy"]).to_excel("Model_Accuracy.xlsx")
