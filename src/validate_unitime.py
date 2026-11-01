
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_csv(filename):
    """Load a CSV file from the processed data directory."""
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    return pd.read_csv(path)


def main():
    print("=" * 60)
    print("UniSched-AI: Benchmark Data Validation")
    print("=" * 60)

    # Load processed UniTime benchmark datasets.
    classes = load_csv("unitime_classes.csv")
    candidates = load_csv("unitime_candidate_assignments.csv")
    rooms = load_csv("unitime_rooms.csv")
    enrollments = load_csv("unitime_student_class_enrollments.csv")
    constraints = load_csv("unitime_group_constraints.csv")

    checks = []

    def check(name, passed, details=""):
        passed = bool(passed)
        checks.append(passed)

        status = "PASS" if passed else "CHECK"
        print(f"[{status}] {name}")

        if details:
            print(f"       {details}")

    # ---------------------------------------------------------
    # 1. Validate class and room identifiers.
    # ---------------------------------------------------------
    check(
        "Class IDs are unique",
        classes["class_id"].is_unique,
    )

    check(
        "Room IDs are unique",
        rooms["room_id"].is_unique,
    )

    check(
        "Class IDs are not missing",
        classes["class_id"].notna().all(),
    )

    # ---------------------------------------------------------
    # 2. Validate candidate assignments.
    # ---------------------------------------------------------
    check(
        "Candidate assignments reference known classes",
        candidates["class_id"].isin(classes["class_id"]).all(),
    )

    check(
        "Candidate assignments reference known rooms",
        candidates["room_id"].isin(rooms["room_id"]).all(),
    )

    check(
        "Candidate durations are positive",
        (candidates["duration_minutes"] > 0).all(),
    )

    check(
        "Candidate end times follow start times",
        (candidates["end_minute"] > candidates["start_minute"]).all(),
    )

    # ---------------------------------------------------------
    # 3. Validate student-class enrollment links.
    # ---------------------------------------------------------
    check(
        "Student links reference known classes",
        enrollments["class_id"].isin(classes["class_id"]).all(),
    )

    # ---------------------------------------------------------
    # 4. Validate group constraint references.
    # The CSV stores one class_id per row, not a class_ids list.
    # ---------------------------------------------------------
    known_classes = set(classes["class_id"].dropna().astype(str))

    constraint_class_ids = (
        constraints["class_id"].dropna().astype(str)
    )

    unknown_constraint_classes = (
        set(constraint_class_ids) - known_classes
    )

    check(
        "Group constraints reference known classes",
        len(unknown_constraint_classes) == 0,
        (
            f"Unknown IDs: "
            f"{sorted(unknown_constraint_classes)[:10]}"
            if unknown_constraint_classes
            else ""
        ),
    )

    check(
        "Group constraint references are not missing",
        constraints["class_id"].notna().all(),
    )

    check(
        "Group constraint types are not missing",
        constraints["constraint_type"].notna().all(),
    )

    # ---------------------------------------------------------
    # 5. Check that candidate option keys are unique.
    # ---------------------------------------------------------
    required_candidate_key_columns = [
        "class_id",
        "time_option_id",
        "room_option_id",
    ]

    missing_key_columns = [
        column
        for column in required_candidate_key_columns
        if column not in candidates.columns
    ]

    if missing_key_columns:
        check(
            "Candidate option keys are unique",
            False,
            f"Missing columns: {missing_key_columns}",
        )
    else:
        duplicate_candidates = candidates.duplicated(
            subset=required_candidate_key_columns
        ).sum()

        check(
            "Candidate option keys are unique",
            duplicate_candidates == 0,
            f"Duplicate keys: {duplicate_candidates}",
        )

    # ---------------------------------------------------------
    # 6. Print validation summary.
    # ---------------------------------------------------------
    print("\n" + "=" * 60)

    passed_count = sum(checks)
    total_checks = len(checks)

    print(f"Checks passed: {passed_count}/{total_checks}")
    print("=" * 60)

    if not all(checks):
        print(
            "Some checks need investigation before this dataset "
            "is used for model training."
        )
    else:
        print(
            "All validation checks passed. This confirms structural "
            "consistency, not timetable feasibility or model accuracy."
        )


if __name__ == "__main__":
    main()
