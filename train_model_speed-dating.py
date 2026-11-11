import joblib
import pandas as pd
from scipy.io import arff
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data, meta = arff.loadarff("data/speeddating.arff")
df = pd.DataFrame(data)

# Target: match (1 if both people said yes)
y = df["match"].str.decode("utf-8").astype(int)

features = [
    "like",
    "attractive_partner",
    "funny_partner",
    "shared_interests_partner",
    "guess_prob_liked",
    "attractive_o",
    "funny_o",
]
X = df[features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=77, stratify=y
)

# Median imputation + scaling + logistic regression balanced for the rare match class
model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(class_weight="balanced", max_iter=1000),
)
model.fit(X_train, y_train)

# Evaluate with ROC AUC and per-class precision/recall
print("ROC AUC:", round(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]), 3))
print(classification_report(y_test, model.predict(X_test)))

# Save the trained pipeline for the API
joblib.dump(model, "model_speed_dating.joblib")
print("Model saved to model_speed_dating.joblib")
