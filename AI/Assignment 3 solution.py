import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import SelectKBest, f_classif

np.random.seed(42)

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

id_col = next((c for c in train.columns if c.lower() in ("id", "row_id")), None)
target_col = [c for c in train.columns if c not in test.columns and c != id_col][0]

if id_col:
    test_ids = test[id_col].copy()
    train = train.drop(columns=[id_col])
    test  = test.drop(columns=[id_col])
else:
    test_ids = range(len(test))

X = train.drop(columns=[target_col])
y = train[target_col].copy()

print(f"Train: {train.shape} | Target: {target_col}")
print("Target distribution:\n", train[target_col].value_counts())

le = LabelEncoder()
y_enc = le.fit_transform(y)

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

for col in cat_cols:
    enc = LabelEncoder()
    combined = pd.concat([X[col], test[col]]).astype(str)
    enc.fit(combined)
    X[col]    = enc.transform(X[col].astype(str))
    test[col] = enc.transform(test[col].astype(str))

X[num_cols]    = X[num_cols].fillna(X[num_cols].median())
test[num_cols] = test[num_cols].fillna(test[num_cols].median())
X[cat_cols]    = X[cat_cols].fillna(-1)
test[cat_cols] = test[cat_cols].fillna(-1)

def create_features(df, num_cols, cat_cols, is_train=True, poly_fit=None):
    df = df.copy()

    top_num = num_cols[:5] if len(num_cols) > 5 else num_cols
    if len(top_num) > 0:
        if is_train:
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            poly_arr = poly.fit_transform(df[top_num])
        else:
            poly_arr = poly_fit.transform(df[top_num])

        poly_cols = [f"poly_{i}" for i in range(poly_arr.shape[1])]
        poly_df   = pd.DataFrame(poly_arr, columns=poly_cols, index=df.index)
        df        = pd.concat([df, poly_df], axis=1)

    if "Soil_Moisture" in df.columns and "Rainfall" in df.columns:
        df["Moisture_Rain_Ratio"] = df["Soil_Moisture"] / (df["Rainfall"] + 1e-5)

    if "Temperature" in df.columns and "Humidity" in df.columns:
        df["Temp_x_Humidity"] = df["Temperature"] * df["Humidity"]

    if "Soil_pH" in df.columns:
        df["pH_Deviation"] = (df["Soil_pH"] - 7.0).abs()

    all_num = df.select_dtypes(include=[np.number]).columns.tolist()
    df["row_mean"]  = df[all_num].mean(axis=1)
    df["row_std"]   = df[all_num].std(axis=1).fillna(0)
    df["row_range"] = df[all_num].max(axis=1) - df[all_num].min(axis=1)

    if is_train:
        return df, poly
    return df

X_eng, poly_fitter = create_features(X, num_cols, cat_cols, is_train=True)
test_eng           = create_features(test, num_cols, cat_cols, is_train=False, poly_fit=poly_fitter)

print(f"Features: {X.shape[1]} → {X_eng.shape[1]}")

scaler = StandardScaler()
X_sc    = scaler.fit_transform(X_eng)
test_sc = scaler.transform(test_eng)

k_best = min(50, X_sc.shape[1])
selector = SelectKBest(f_classif, k=k_best)
X_sc    = selector.fit_transform(X_sc, y_enc)
test_sc = selector.transform(test_sc)

class LinearClassifier(BaseEstimator, ClassifierMixin):
    """Linear Regression used as a classifier."""
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.model = LinearRegression()
        self.model.fit(X, y)
        return self

    def predict(self, X):
        preds = np.round(self.model.predict(X)).astype(int)
        return np.clip(preds, self.classes_.min(), self.classes_.max())


class KMeansClassifier(BaseEstimator, ClassifierMixin):
    """K-Means clustering mapped to majority class per cluster."""
    def __init__(self, k=3, random_state=42):
        self.k = k
        self.random_state = random_state

    def fit(self, X, y):
        self.km = KMeans(n_clusters=self.k, n_init=10, random_state=self.random_state)
        self.km.fit(X)
        self.label_map = {}
        for c in np.unique(self.km.labels_):
            mask = self.km.labels_ == c
            self.label_map[c] = pd.Series(y[mask]).mode()[0] if mask.sum() > 0 else 0
        return self

    def predict(self, X):
        clusters = self.km.predict(X)
        return np.array([self.label_map.get(c, 0) for c in clusters])


class HardVotingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Manual hard-voting ensemble.
    Avoids sklearn VotingClassifier validation bugs with custom estimators.
    """
    def __init__(self, estimators):
        self.estimators = estimators  

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.fitted_estimators_ = []
        for name, est in self.estimators:
            est_clone = clone(est)
            est_clone.fit(X, y)
            self.fitted_estimators_.append(est_clone)
        return self

    def predict(self, X):
        preds = np.array([est.predict(X) for est in self.fitted_estimators_])
        # Majority vote per sample
        voted = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=len(self.classes_)).argmax(),
            axis=0, arr=preds)
        return voted


n_classes = len(np.unique(y_enc))

all_models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=12, min_samples_split=10, min_samples_leaf=4,
        class_weight='balanced', random_state=42),

    "Naive Bayes": GaussianNB(),

    "Logistic Regression": LogisticRegression(
        max_iter=500, C=0.5, class_weight='balanced',
        random_state=42, n_jobs=-1),

    "Linear Classifier": LinearClassifier(),

    "KMeans": KMeansClassifier(k=n_classes),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def kfold_balanced_accuracy(model, data):
    return cross_val_score(
        model, data, y_enc, cv=cv,
        scoring="balanced_accuracy", n_jobs=-1
    ).mean()

def oof_confusion_matrix(model, data):
    oof  = np.zeros(len(data), dtype=int)
    darr = np.array(data)
    for tr, val in cv.split(darr, y_enc):
        m = clone(model)
        m.fit(darr[tr], y_enc[tr])
        oof[val] = m.predict(darr[val])
    return confusion_matrix(y_enc, oof)

results = {}
print("\nMODEL PERFORMANCE (Balanced Accuracy)\n")

for name, model in all_models.items():
    print(f"[required] {name} ...")
    kf = kfold_balanced_accuracy(model, X_sc)
    cm = oof_confusion_matrix(model, X_sc)
    results[name] = kf
    print(f"  KFold  : {kf:.4f}")
    print(f"  Confusion Matrix (OOF):\n{cm}")
    print("-" * 40)

print("\nBuilding Voting Ensemble ...")
top3 = sorted(results, key=results.get, reverse=True)[:3]
print(f"  Top 3: {top3}")

def fresh_model(name):
    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=12, min_samples_split=10,
                                      min_samples_leaf=4, class_weight='balanced',
                                      random_state=42)
    if name == "Naive Bayes":
        return GaussianNB()
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=500, C=0.5,
                                  class_weight='balanced',
                                  random_state=42, n_jobs=-1)
    if name == "Linear Classifier":
        return LinearClassifier()
    if name == "KMeans":
        return KMeansClassifier(k=n_classes)
    return LogisticRegression(max_iter=500)

ensemble = HardVotingEnsemble(estimators=[(n, fresh_model(n)) for n in top3])
ensemble_cv = kfold_balanced_accuracy(ensemble, X_sc)
results["Voting Ensemble"] = ensemble_cv
print(f"  Ensemble CV Balanced Acc: {ensemble_cv:.4f}")

n_plots = len(all_models) + 1
ncols   = 3
nrows   = (n_plots + ncols - 1)

fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
axes_flat = axes.flatten()

for i, (name, model) in enumerate(all_models.items()):
    ConfusionMatrixDisplay(oof_confusion_matrix(model, X_sc)).plot(
        ax=axes_flat[i], colorbar=False)
    axes_flat[i].set_title(name)

oof_ens = np.zeros(len(X_sc), dtype=int)
for tr, val in cv.split(X_sc, y_enc):
    m = clone(ensemble)
    m.fit(X_sc[tr], y_enc[tr])
    oof_ens[val] = m.predict(X_sc[val])

ConfusionMatrixDisplay(confusion_matrix(y_enc, oof_ens)).plot(
    ax=axes_flat[len(all_models)], colorbar=False)
axes_flat[len(all_models)].set_title("Voting Ensemble")

for ax in axes_flat[len(all_models) + 1:]:
    ax.set_visible(False)

plt.suptitle("Confusion Matrices – All Models (OOF)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved => confusion_matrices.png")

best_name = max(results, key=results.get)
print(f"\nBest model: {best_name}  (CV={results[best_name]:.4f})")

if best_name == "Voting Ensemble":
    best_model = HardVotingEnsemble(estimators=[(n, fresh_model(n)) for n in top3])
    best_model.fit(X_sc, y_enc)
    preds = best_model.predict(test_sc)
else:
    best_model = fresh_model(best_name)
    best_model.fit(X_sc, y_enc)
    preds = best_model.predict(test_sc)

preds = le.inverse_transform(preds.astype(int))

submission = pd.DataFrame({
    (id_col if id_col else "id"): test_ids,
    target_col: preds
})
submission.to_csv("submission.csv", index=False)
print("Saved => submission.csv")
print(submission.head())

print("\n--- Final Balanced Accuracy Scores (sorted) ---")
for k, v in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {k:25s}: {v:.4f}")
print("\nDone!")