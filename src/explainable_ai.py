
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance


# ============================================================
# 1. PROJECT PATHS AND SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = PROJECT_ROOT / "models" / "best_conflict_model.joblib"
DATA_FILE = PROJECT_ROOT / "data" / "ml_timetable_dataset.csv"

XAI_DIR = PROJECT_ROOT / "reports" / "explainable_ai"

PERMUTATION_FILE = XAI_DIR / "permutation_importance.csv"
SHAP_FILE = XAI_DIR / "shap_feature_importance.csv"
SHAP_PLOT_FILE = XAI_DIR / "shap_summary.png"

RANDOM_STATE = 42

CATEGORICAL_FEATURES = [
    "day",
    "room",
    "faculty",
    "student_group",
]

FEATURE_COLUMNS = [
    "day",
    "room",
    "faculty",
    "student_group",
    "duration_minutes",
    "start_time_sin",
    "start_time_cos",
    "overlapping_room_classes",
    "overlapping_faculty_classes",
    "overlapping_group_classes",
    "overlapping_resource_classes",
]


# ============================================================
# 2. LOAD MODEL AND DATASET
# ============================================================

def load_model_and_data():
    """Load the saved model and ML dataset."""

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}\n"
            "Run: python src/train_model.py"
        )

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    model = joblib.load(MODEL_FILE)
    df = pd.read_csv(DATA_FILE)

    return model, df


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

def prepare_explanation_data(df):
    """
    Recreate the same features used during model training.

    Every schedule is processed independently.
    """

    data = df.copy()

    def to_minutes(value):
        hour, minute = map(int, str(value).split(":"))
        return hour * 60 + minute

    data["start_minutes"] = data["start_time"].map(to_minutes)
    data["end_minutes"] = data["end_time"].map(to_minutes)

    if (data["end_minutes"] <= data["start_minutes"]).any():
        raise ValueError(
            "Every class must end after it starts."
        )

    # Cyclical time features.
    data["start_time_sin"] = np.sin(
        2 * np.pi * data["start_minutes"] / 1440
    )

    data["start_time_cos"] = np.cos(
        2 * np.pi * data["start_minutes"] / 1440
    )

    # Initialize overlap features.
    data["overlapping_room_classes"] = 0
    data["overlapping_faculty_classes"] = 0
    data["overlapping_group_classes"] = 0
    data["overlapping_resource_classes"] = 0

    # Process each schedule independently.
    for schedule_id, schedule in data.groupby("schedule_id"):

        print(f"Preparing features for {schedule_id}...")

        indices = schedule.index.to_numpy()

        days = schedule["day"].to_numpy()

        starts = data.loc[
            indices, "start_minutes"
        ].to_numpy()

        ends = data.loc[
            indices, "end_minutes"
        ].to_numpy()

        rooms = schedule["room"].to_numpy()
        faculty = schedule["faculty"].to_numpy()
        groups = schedule["student_group"].to_numpy()

        n = len(schedule)

        room_counts = np.zeros(n, dtype=int)
        faculty_counts = np.zeros(n, dtype=int)
        group_counts = np.zeros(n, dtype=int)
        total_counts = np.zeros(n, dtype=int)

        for i in range(n):

            same_day = days == days[i]

            overlaps = (
                (starts < ends[i])
                & (ends > starts[i])
            )

            other_classes = np.arange(n) != i

            candidates = (
                same_day
                & overlaps
                & other_classes
            )

            shared_room = (
                (rooms == rooms[i])
                & candidates
            )

            shared_faculty = (
                (faculty == faculty[i])
                & candidates
            )

            shared_group = (
                (groups == groups[i])
                & candidates
            )

            room_counts[i] = int(shared_room.sum())
            faculty_counts[i] = int(shared_faculty.sum())
            group_counts[i] = int(shared_group.sum())

            shared_resource = (
                shared_room
                | shared_faculty
                | shared_group
            )

            total_counts[i] = int(shared_resource.sum())

        data.loc[
            indices, "overlapping_room_classes"
        ] = room_counts

        data.loc[
            indices, "overlapping_faculty_classes"
        ] = faculty_counts

        data.loc[
            indices, "overlapping_group_classes"
        ] = group_counts

        data.loc[
            indices, "overlapping_resource_classes"
        ] = total_counts

    X = data[FEATURE_COLUMNS].copy()
    y = data["has_conflict"].astype(int)

    return X, y


# ============================================================
# 4. PERMUTATION FEATURE IMPORTANCE
# ============================================================

def calculate_permutation_importance(model, X, y):
    """
    Measure the decrease in F1 score when individual
    features are shuffled.
    """

    print("\nCalculating permutation feature importance...")

    sample_size = min(1000, len(X))

    X_sample = X.sample(
        n=sample_size,
        random_state=RANDOM_STATE
    )

    y_sample = y.loc[X_sample.index]

    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        n_repeats=5,
        random_state=RANDOM_STATE,
        scoring="f1",
        n_jobs=1
    )

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })

    importance_df = importance_df.sort_values(
        by="importance_mean",
        ascending=False
    ).reset_index(drop=True)

    return importance_df


# ============================================================
# 5. NUMERICAL ENCODING FOR SHAP
# ============================================================

def encode_categorical_features(X):
    """
    Encode categorical values as numerical codes for SHAP.

    Keep reverse mappings so predictions can use the
    original categorical values expected by the pipeline.
    """

    X_numeric = X.copy()

    category_mappings = {}

    for feature in CATEGORICAL_FEATURES:

        categories = sorted(
            X[feature].astype(str).unique()
        )

        category_to_code = {
            category: index
            for index, category in enumerate(categories)
        }

        code_to_category = {
            index: category
            for category, index in category_to_code.items()
        }

        category_mappings[feature] = code_to_category

        X_numeric[feature] = (
            X_numeric[feature]
            .astype(str)
            .map(category_to_code)
        )

    X_numeric = X_numeric.astype(float)

    return X_numeric, category_mappings


# ============================================================
# 6. SHAP EXPLANATIONS
# ============================================================

def calculate_shap_importance(model, X):
    """
    Calculate SHAP values using a permutation explainer.

    SHAP receives numerical feature values.
    The prediction function restores categorical values
    before passing records to the original ML pipeline.
    """

    print("\nCalculating SHAP explanations...")

    feature_names = list(X.columns)

    # Encode categorical features.
    X_numeric, category_mappings = (
        encode_categorical_features(X)
    )

    # Select explanation and background samples.
    X_sample = X_numeric.sample(
        n=min(100, len(X_numeric)),
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    background = X_numeric.sample(
        n=min(30, len(X_numeric)),
        random_state=24
    ).reset_index(drop=True)

    # Prediction function used by SHAP.
    def predict_function(values):

        input_df = pd.DataFrame(
            values,
            columns=feature_names
        )

        # Restore original categorical values.
        for feature in CATEGORICAL_FEATURES:

            reverse_mapping = category_mappings[feature]

            codes = (
                input_df[feature]
                .round()
                .astype(int)
                .clip(
                    lower=min(reverse_mapping),
                    upper=max(reverse_mapping)
                )
            )

            input_df[feature] = codes.map(reverse_mapping)

        # Return probability of conflict.
        return model.predict_proba(input_df)[:, 1]

    # Numerical masker prevents string subtraction errors.
    masker = shap.maskers.Independent(
        background.to_numpy(dtype=float),
        max_samples=len(background)
    )

    explainer = shap.Explainer(
        predict_function,
        masker,
        algorithm="permutation",
        feature_names=feature_names
    )

    print("Explaining sampled timetable records...")

    shap_values = explainer(
        X_sample.to_numpy(dtype=float),
        max_evals=2 * len(feature_names) + 1
    )

    values = np.asarray(shap_values.values)

    # Handle multi-output SHAP results if returned.
    if values.ndim == 3:
        values = values[:, :, -1]

    # Calculate mean absolute SHAP value.
    mean_importance = np.abs(values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_absolute_shap_value": mean_importance,
    })

    importance_df = importance_df.sort_values(
        by="mean_absolute_shap_value",
        ascending=False
    ).reset_index(drop=True)

    return shap_values, importance_df


# ============================================================
# 7. SAVE SHAP SUMMARY PLOT
# ============================================================

def save_shap_plot(shap_values):
    """
    Save a SHAP global feature-importance bar plot.

    A bar plot is used because categorical features are
    internally encoded as numbers; a beeswarm color gradient
    could misleadingly suggest that category codes have order.
    """

    print("\nCreating SHAP summary plot...")

    plt.figure(figsize=(12, 7))

    shap.plots.bar(
        shap_values,
        max_display=len(FEATURE_COLUMNS),
        show=False
    )

    plt.title(
        "UniSched-AI: SHAP Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        SHAP_PLOT_FILE,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 8. MAIN EXECUTION
# ============================================================

def main():

    print("=" * 65)
    print("UniSched-AI: Explainable AI Pipeline")
    print("=" * 65)

    XAI_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load model and data.
    model, df = load_model_and_data()

    print(f"\nDataset shape: {df.shape}")

    # Prepare features.
    X, y = prepare_explanation_data(df)

    print(f"\nFeature shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")

    # Calculate permutation importance.
    permutation_df = calculate_permutation_importance(
        model,
        X,
        y
    )

    permutation_df.to_csv(
        PERMUTATION_FILE,
        index=False
    )

    # Calculate SHAP importance.
    shap_values, shap_df = calculate_shap_importance(
        model,
        X
    )

    shap_df.to_csv(
        SHAP_FILE,
        index=False
    )

    # Save SHAP plot.
    save_shap_plot(shap_values)

    # Display permutation results.
    print("\n" + "=" * 65)
    print("PERMUTATION FEATURE IMPORTANCE")
    print("=" * 65)

    print(
        permutation_df.to_string(index=False)
    )

    # Display SHAP results.
    print("\n" + "=" * 65)
    print("SHAP FEATURE IMPORTANCE")
    print("=" * 65)

    print(
        shap_df.to_string(index=False)
    )

    print("\n" + "=" * 65)
    print("XAI PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 65)

    print("\nGenerated files:")

    print(f"1. {PERMUTATION_FILE}")
    print(f"2. {SHAP_FILE}")
    print(f"3. {SHAP_PLOT_FILE}")

    print("\nImportant limitations:")
    print(
        "1. The dataset contains synthetic timetable records."
    )
    print(
        "2. SHAP explains model behavior, not causal relationships."
    )
    print(
        "3. Categorical values are encoded internally for SHAP."
    )
    print(
        "4. Results should not be interpreted as proof of "
        "real-world predictive performance."
    )


if __name__ == "__main__":
    main()
