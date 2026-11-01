
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


# ============================================================
# 1. PROJECT PATHS AND SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "ml_timetable_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODEL_DIR / "best_conflict_model.joblib"
RESULTS_FILE = PROJECT_ROOT / "data" / "model_evaluation.csv"

TARGET = "has_conflict"
GROUP_COLUMN = "schedule_id"
RANDOM_STATE = 42


# ============================================================
# 2. LOAD AND VALIDATE DATASET
# ============================================================

def load_dataset():
    """Load the ML dataset and validate its structure."""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Run: python src/generate_ml_dataset.py"
        )

    df = pd.read_csv(DATA_FILE)

    required_columns = {
        GROUP_COLUMN,
        "class_id",
        "day",
        "start_time",
        "end_time",
        "room",
        "faculty",
        "student_group",
        "duration_minutes",
        TARGET,
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("The dataset is empty.")

    if df[list(required_columns)].isna().any().any():
        raise ValueError(
            "Required columns contain missing values."
        )

    if set(df[TARGET].unique()) != {0, 1}:
        raise ValueError(
            "The target column must contain both labels: 0 and 1."
        )

    if df["class_id"].duplicated().any():
        raise ValueError("Duplicate class IDs found.")

    if df[GROUP_COLUMN].nunique() < 2:
        raise ValueError(
            "At least two different schedules are required."
        )

    return df


# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

def prepare_features(df):
    """
    Calculate timetable conflict features independently
    within each schedule.

    This prevents one schedule from affecting another.
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

    # Initialize conflict feature columns.
    data["overlapping_room_classes"] = 0
    data["overlapping_faculty_classes"] = 0
    data["overlapping_group_classes"] = 0
    data["overlapping_resource_classes"] = 0

    # Process each timetable independently.
    for schedule_id, schedule in data.groupby(GROUP_COLUMN):

        print(f"Calculating features for {schedule_id}...")

        indices = schedule.index.to_numpy()

        days = schedule["day"].to_numpy()
        starts = schedule["start_minutes"].to_numpy()
        ends = schedule["end_minutes"].to_numpy()

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

        # Store features in the original dataframe.
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

    feature_columns = [
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

    X = data[feature_columns].copy()
    y = data[TARGET].astype(int)
    groups = data[GROUP_COLUMN].copy()

    return X, y, groups


# ============================================================
# 4. PREPROCESSING PIPELINE
# ============================================================

def build_preprocessor():
    """Encode categorical features and scale numerical features."""

    categorical_features = [
        "day",
        "room",
        "faculty",
        "student_group",
    ]

    numerical_features = [
        "duration_minutes",
        "start_time_sin",
        "start_time_cos",
        "overlapping_room_classes",
        "overlapping_faculty_classes",
        "overlapping_group_classes",
        "overlapping_resource_classes",
    ]

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
            (
                "numerical",
                numerical_pipeline,
                numerical_features,
            ),
        ]
    )


# ============================================================
# 5. DEFINE AND COMPARE MODELS
# ============================================================

def train_models(X_train, X_test, y_train, y_test):

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = []
    trained_pipelines = {}

    for model_name, classifier in models.items():

        print("\n" + "=" * 60)
        print(f"Training {model_name}")
        print("=" * 60)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", classifier),
            ]
        )

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        metrics = {
            "model": model_name,
            "accuracy": accuracy_score(
                y_test, predictions
            ),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1_score": f1_score(
                y_test, predictions, zero_division=0
            ),
        }

        results.append(metrics)
        trained_pipelines[model_name] = pipeline

        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                predictions,
                target_names=["No Conflict", "Conflict"],
                zero_division=0,
            )
        )

        print("Confusion Matrix:")
        print(
            confusion_matrix(y_test, predictions)
        )

    results_df = pd.DataFrame(results).sort_values(
        by="f1_score",
        ascending=False,
    ).reset_index(drop=True)

    best_model_name = results_df.iloc[0]["model"]

    best_pipeline = trained_pipelines[best_model_name]

    return results_df, best_model_name, best_pipeline


# ============================================================
# 6. MAIN EXECUTION
# ============================================================

def main():

    print("=" * 60)
    print("UniSched-AI: ML Training Pipeline")
    print("=" * 60)

    df = load_dataset()

    print(f"\nDataset shape: {df.shape}")

    print("\nTarget distribution:")
    print(
        df[TARGET]
        .value_counts()
        .sort_index()
        .rename(
            index={
                0: "No Conflict",
                1: "Conflict",
            }
        )
    )

    # Generate features separately for every schedule.
    X, y, groups = prepare_features(df)

    print(f"\nFeature count: {X.shape[1]}")

    print("\nFeatures used:")
    for feature in X.columns:
        print(f"- {feature}")

    # Split by complete schedules, not individual rows.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    train_indices, test_indices = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]

    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    train_groups = set(groups.iloc[train_indices])
    test_groups = set(groups.iloc[test_indices])

    print("\nSchedule-based split:")
    print(f"Training schedules: {len(train_groups)}")
    print(f"Testing schedules: {len(test_groups)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    # Verify that no schedule appears in both sets.
    overlap = train_groups.intersection(test_groups)

    if overlap:
        raise RuntimeError(
            f"Data leakage detected! Shared schedules: {overlap}"
        )

    print("\nNo schedule overlap between training and testing.")

    # Verify both target classes exist in both partitions.
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(
            "Both training and testing sets must contain "
            "both conflict classes. Try another random seed."
        )

    results_df, best_name, best_pipeline = train_models(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # Save the trained model and evaluation results.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_pipeline, MODEL_FILE)

    results_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(results_df.to_string(index=False))

    print(f"\nBest model by F1-score: {best_name}")
    print(f"Saved model: {MODEL_FILE}")
    print(f"Saved evaluation results: {RESULTS_FILE}")

    print("\nIMPORTANT LIMITATIONS:")
    print(
        "1. This dataset is synthetic, not real university data."
    )
    print(
        "2. Conflict features are calculated from complete "
        "individual schedules."
    )
    print(
        "3. Group-based splitting prevents the same schedule "
        "from appearing in training and testing."
    )
    print(
        "4. High scores may reflect the synthetic data-generation "
        "rules and do not prove real-world predictive performance."
    )


if __name__ == "__main__":
    main()
