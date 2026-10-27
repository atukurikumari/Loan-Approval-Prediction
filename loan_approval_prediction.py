# ==============================================================
# LOAN APPROVAL PREDICTION - COMPLETE CASE STUDY
# ==============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    precision_recall_curve
)

# ==============================================================
# 1. LOAD DATASET
# ==============================================================

DATA_PATH = (
    "/kaggle/input/datasets/bhanupratapbiswas/"
    "loan-approval-prediction-case-study/loan_prediction.csv"
)

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("LOAN APPROVAL PREDICTION")
print("=" * 60)

print("\nDataset Shape:", df.shape)
print("\nFirst Five Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

# ==============================================================
# 2. CLEAN DATA
# ==============================================================

df.columns = df.columns.str.strip()
df = df.drop_duplicates().copy()

# Automatically identify the target column
target_mapping = {
    "loanstatus": "loan_status",
    "loanapproval": "loan_approval",
    "approvalstatus": "approval_status",
    "loanapproved": "loan_approved"
}

column_lookup = {
    col.lower().replace("_", "").replace(" ", "").replace("-", ""): col
    for col in df.columns
}

target_col = None

for key in target_mapping:
    if key in column_lookup:
        target_col = column_lookup[key]
        break

if target_col is None:
    raise ValueError(
        "Loan target column not found. Available columns: "
        + str(df.columns.tolist())
        + ". Set target_col to the correct column name."
    )

print("\nTarget Column:", target_col)

# Remove records with missing target values
df = df.dropna(subset=[target_col]).copy()

# ==============================================================
# 3. CONVERT TARGET INTO BINARY FORMAT
# 1 = APPROVED
# 0 = REJECTED
# ==============================================================

target = df[target_col]

if pd.api.types.is_numeric_dtype(target):

    unique_values = set(target.unique())

    if unique_values.issubset({0, 1}):
        y = target.astype(int)
    else:
        raise ValueError(
            "Numeric target must contain 0 and 1. "
            "Define the target mapping for your dataset."
        )

else:

    target_text = target.astype(str).str.strip().str.lower()

    approved_labels = {
        "y", "yes", "approved", "approve",
        "accepted", "true", "1"
    }

    rejected_labels = {
        "n", "no", "rejected", "reject",
        "declined", "false", "0"
    }

    labels = set(target_text.unique())

    if not labels.issubset(approved_labels | rejected_labels):
        raise ValueError(
            "Unknown target labels: " + str(labels)
            + ". Update approved_labels and rejected_labels."
        )

    y = target_text.isin(approved_labels).astype(int)

print("\nTarget Distribution:")
print(y.value_counts())

print("\nTarget Percentages:")
print((y.value_counts(normalize=True) * 100).round(2))

if y.nunique() != 2 or y.value_counts().min() < 2:
    raise ValueError(
        "The dataset needs both target classes and at least "
        "two examples of each class."
    )

# ==============================================================
# 4. PREPARE FEATURES
# ==============================================================

X = df.drop(columns=[target_col]).copy()

# Remove identifier columns
id_columns = [
    col for col in X.columns
    if col.lower().replace("_", "").replace(" ", "")
    in {
        "id",
        "loanid",
        "applicationid",
        "applicantid"
    }
]

X = X.drop(columns=id_columns)

# Remove constant columns
constant_columns = [
    col for col in X.columns
    if X[col].nunique(dropna=False) <= 1
]

X = X.drop(columns=constant_columns)

# Convert Boolean columns into integers
for col in X.select_dtypes(include=["bool"]).columns:
    X[col] = X[col].astype(int)

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

print("\nNumerical Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)

# ==============================================================
# 5. PREPROCESSING PIPELINE
# ==============================================================

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# ==============================================================
# 6. SPLIT DATA
# ==============================================================

# 60% training, 20% validation, 20% testing

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.25,
    random_state=42,
    stratify=y_train_full
)

print("\nTraining Records:", len(X_train))
print("Validation Records:", len(X_val))
print("Testing Records:", len(X_test))

# ==============================================================
# 7. DEFINE MACHINE LEARNING MODELS
# ==============================================================

models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=8,
        min_samples_leaf=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
}

# ==============================================================
# 8. TRAIN MODELS
# ==============================================================

trained_models = {}
validation_probs = {}
test_probs = {}
results = []

for name, classifier in models.items():

    print("\nTraining:", name)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    val_probability = pipeline.predict_proba(X_val)[:, 1]
    test_probability = pipeline.predict_proba(X_test)[:, 1]

    trained_models[name] = pipeline
    validation_probs[name] = val_probability
    test_probs[name] = test_probability

    # Default threshold
    test_prediction = (test_probability >= 0.50).astype(int)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, test_prediction),
        "Precision": precision_score(
            y_test, test_prediction, zero_division=0
        ),
        "Recall": recall_score(
            y_test, test_prediction, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, test_prediction, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, test_probability)
    })

# ==============================================================
# 9. COMPARE MODEL PERFORMANCE
# ==============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="ROC-AUC",
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    "loan_model_comparison.csv",
    index=False
)

# ==============================================================
# 10. PLOT MODEL COMPARISON
# ==============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]

results_df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Loan Approval - Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# ==============================================================
# 11. SELECT THRESHOLD USING VALIDATION DATA
# ==============================================================

threshold_results = []

for name, probabilities in validation_probs.items():

    precision, recall, thresholds = precision_recall_curve(
        y_val,
        probabilities
    )

    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-10)
    )

    best_index = int(np.argmax(f1_values))

    threshold_results.append({
        "Model": name,
        "Threshold": float(thresholds[best_index]),
        "Validation F1": float(f1_values[best_index])
    })

threshold_df = pd.DataFrame(threshold_results)

print("\nTHRESHOLD COMPARISON")
print(threshold_df.round(4).to_string(index=False))

threshold_df.to_csv(
    "loan_threshold_comparison.csv",
    index=False
)

# Choose the model with the highest validation F1
best_row = threshold_df.loc[
    threshold_df["Validation F1"].idxmax()
]

best_model_name = best_row["Model"]
best_threshold = float(best_row["Threshold"])

best_model = trained_models[best_model_name]
best_probability = test_probs[best_model_name]

print("\nSelected Model:", best_model_name)
print("Selected Threshold:", round(best_threshold, 4))

# ==============================================================
# 12. FINAL EVALUATION ON TEST DATA
# ==============================================================

y_pred = (best_probability >= best_threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, best_probability)

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print("Model:", best_model_name)
print("Threshold:", round(best_threshold, 4))
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("ROC-AUC:", round(roc_auc, 4))

print("\nCLASSIFICATION REPORT")

print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Rejected", "Approved"],
    zero_division=0
))

# ==============================================================
# 13. CONFUSION MATRIX
# ==============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Rejected", "Approved"]
)

disp.plot(values_format="d")

plt.title("Confusion Matrix - " + best_model_name)
plt.tight_layout()
plt.show()

tn, fp, fn, tp = cm.ravel()

print("\nCONFUSION MATRIX INTERPRETATION")
print("Correctly predicted rejections:", tn)
print("Rejected cases predicted approved:", fp)
print("Approved cases predicted rejected:", fn)
print("Correctly predicted approvals:", tp)

# ==============================================================
# 14. ROC CURVE
# ==============================================================

plt.figure(figsize=(9, 6))

for name, probabilities in test_probs.items():

    RocCurveDisplay.from_predictions(
        y_test,
        probabilities,
        name=name,
        ax=plt.gca()
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ROC Curve - Model Comparison")
plt.tight_layout()
plt.show()

# ==============================================================
# 15. FEATURE IMPORTANCE
# ==============================================================

fitted_preprocessor = best_model.named_steps["preprocessor"]
fitted_classifier = best_model.named_steps["classifier"]

feature_names = fitted_preprocessor.get_feature_names_out()

if hasattr(fitted_classifier, "feature_importances_"):

    importance_values = fitted_classifier.feature_importances_

elif hasattr(fitted_classifier, "coef_"):

    importance_values = np.abs(
        fitted_classifier.coef_[0]
    )

else:

    importance_values = None

if importance_values is not None:

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance_values
    }).sort_values(
        by="Importance",
        ascending=False
    )

    print("\nTOP 15 IMPORTANT FEATURES")

    print(
        importance_df.head(15).to_string(index=False)
    )

    top_features = importance_df.head(15).sort_values(
        by="Importance",
        ascending=True
    )

    top_features.plot(
        x="Feature",
        y="Importance",
        kind="barh",
        figsize=(10, 7),
        legend=False
    )

    plt.title("Top 15 Features")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

# ==============================================================
# 16. BUSINESS INTERPRETATION
# ==============================================================

print("\n" + "=" * 60)
print("BUSINESS INTERPRETATION")
print("=" * 60)

print(
    f"\nThe selected model is {best_model_name}."
)

print(
    f"Test ROC-AUC: {roc_auc:.4f}"
)

print(
    f"Test precision: {precision:.4f}"
)

print(
    f"Test recall: {recall:.4f}"
)

print(
    f"Test F1-score: {f1:.4f}"
)

print("\nBusiness Recommendations:")

print(
    "1. Precision measures how many predicted approvals "
    "are actually approved in the historical data."
)

print(
    "2. Recall measures how many actual approvals "
    "the model correctly identifies."
)

print(
    "3. False approvals and false rejections have "
    "different financial consequences."
)

print(
    "4. Select a deployment threshold using expected "
    "credit losses, missed applicants, and business costs."
)

print(
    "5. Validate fairness, regulatory compliance, and "
    "model performance before deployment."
)

print(
    "6. Predictions should support human review rather "
    "than automatically determine lending decisions."
)

# ==============================================================
# 17. SAVE PREDICTIONS
# ==============================================================

prediction_output = pd.DataFrame({
    "Actual_Status": y_test.to_numpy(),
    "Predicted_Status": y_pred,
    "Approval_Probability": best_probability,
    "Selected_Threshold": best_threshold
})

prediction_output["Predicted_Label"] = np.where(
    prediction_output["Predicted_Status"] == 1,
    "Approved",
    "Rejected"
)

prediction_output.to_csv(
    "loan_test_predictions.csv",
    index=False
)

print("\nFiles created successfully:")
print("1. loan_model_comparison.csv")
print("2. loan_threshold_comparison.csv")
print("3. loan_test_predictions.csv")

print("\nLOAN APPROVAL PREDICTION COMPLETED!")
