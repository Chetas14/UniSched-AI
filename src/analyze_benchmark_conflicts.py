
from pathlib import Path
from itertools import combinations

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"

ASSIGNMENTS_FILE = "unitime_selected_assignments.csv"
CLASSES_FILE = "unitime_classes.csv"
ENROLLMENTS_FILE = "unitime_student_class_enrollments.csv"


def load_data():
    """Load selected assignments and supporting benchmark data."""
    assignments = pd.read_csv(
        DATA_DIR / ASSIGNMENTS_FILE,
        dtype={
            "class_id": str,
            "room_id": str,
            "days_mask": str,
            "dates_mask": str,
        },
    )

    classes = pd.read_csv(
        DATA_DIR / CLASSES_FILE,
        dtype={
            "class_id": str,
            "instructor_ids": str,
        },
    )

    enrollments = pd.read_csv(
        DATA_DIR / ENROLLMENTS_FILE,
        dtype={
            "student_id": str,
            "class_id": str,
        },
    )

    return assignments, classes, enrollments


def masks_overlap(mask_a, mask_b):
    """
    Check whether two binary masks have an active bit
    at the same calendar position.

    UniTime date masks in this dataset can have different
    lengths. This implementation preserves their beginnings
    and pads the ends of shorter masks.
    """
    if pd.isna(mask_a) or pd.isna(mask_b):
        return False

    a = str(mask_a).strip()
    b = str(mask_b).strip()

    if not a or not b:
        return False

    if set(a) - {"0", "1"} or set(b) - {"0", "1"}:
        raise ValueError("A date/day mask contains characters other than 0 and 1.")

    width = max(len(a), len(b))

    # Align from the first position, not the last position.
    a = a.ljust(width, "0")
    b = b.ljust(width, "0")

    return any(
        bit_a == "1" and bit_b == "1"
        for bit_a, bit_b in zip(a, b)
    )


def schedules_overlap(a, b):
    """Check date, day, and time overlap between two classes."""
    if not masks_overlap(a["dates_mask"], b["dates_mask"]):
        return False

    if not masks_overlap(a["days_mask"], b["days_mask"]):
        return False

    # Half-open intervals: a class ending at 10:00 does not
    # overlap a class starting at 10:00.
    return (
        max(a["start_minute"], b["start_minute"])
        < min(a["end_minute"], b["end_minute"])
    )


def build_student_lookup(enrollments):
    """Map each class ID to its enrolled student IDs."""
    result = {}

    for class_id, group in enrollments.groupby("class_id"):
        result[str(class_id)] = set(
            group["student_id"].dropna().astype(str)
        )

    return result


def build_instructor_lookup(classes):
    """Map each class ID to its recorded instructor IDs."""
    result = {}

    for _, row in classes.iterrows():
        class_id = str(row["class_id"])
        value = row.get("instructor_ids")

        if pd.isna(value) or not str(value).strip():
            result[class_id] = set()
            continue

        normalized = str(value).replace(";", ",")

        result[class_id] = {
            instructor.strip()
            for instructor in normalized.split(",")
            if instructor.strip()
        }

    return result


def consolidate_assignments(assignments):
    """
    Consolidate room rows into one timetable record per class.

    Some classes have multiple selected room assignments.
    Their shared timetable information must be consistent.
    """
    required_columns = {
        "class_id",
        "room_id",
        "days_mask",
        "dates_mask",
        "start_minute",
        "end_minute",
    }

    missing_columns = required_columns - set(assignments.columns)

    if missing_columns:
        raise ValueError(
            "Selected assignments are missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if assignments["class_id"].isna().any():
        raise ValueError("Selected assignments contain missing class IDs.")

    assignments = assignments.copy()

    for column in ["start_minute", "end_minute"]:
        assignments[column] = pd.to_numeric(
            assignments[column],
            errors="raise",
        )

    if (
        assignments["end_minute"] <= assignments["start_minute"]
    ).any():
        raise ValueError(
            "Found assignments where end time is not after start time."
        )

    schedules = []

    for class_id, group in assignments.groupby(
        "class_id",
        sort=False,
    ):
        # Every room row for a class must describe the same timetable.
        for column in [
            "days_mask",
            "dates_mask",
            "start_minute",
            "end_minute",
        ]:
            if group[column].nunique(dropna=False) != 1:
                raise ValueError(
                    f"Class {class_id} has inconsistent {column} values."
                )

        first = group.iloc[0]

        room_ids = {
            str(room_id).strip()
            for room_id in group["room_id"].dropna()
            if str(room_id).strip()
        }

        schedules.append(
            {
                "class_id": str(class_id),
                "room_ids": room_ids,
                "days_mask": first["days_mask"],
                "dates_mask": first["dates_mask"],
                "start_minute": int(first["start_minute"]),
                "end_minute": int(first["end_minute"]),
                "source_assignment_rows": len(group),
            }
        )

    return schedules


def analyze_conflicts(assignments, classes, enrollments):
    """Detect room, shared-student, and shared-instructor conflicts."""
    student_lookup = build_student_lookup(enrollments)
    instructor_lookup = build_instructor_lookup(classes)

    schedules = consolidate_assignments(assignments)

    conflict_records = []

    room_conflict_count = 0
    student_conflict_count = 0
    instructor_conflict_count = 0
    overlapping_pairs_checked = 0

    for a, b in combinations(schedules, 2):
        class_a = a["class_id"]
        class_b = b["class_id"]

        # Each record now represents a unique class.
        if class_a == class_b:
            continue

        if not schedules_overlap(a, b):
            continue

        overlapping_pairs_checked += 1

        shared_rooms = a["room_ids"].intersection(b["room_ids"])

        students_a = student_lookup.get(class_a, set())
        students_b = student_lookup.get(class_b, set())
        shared_students = students_a.intersection(students_b)

        instructors_a = instructor_lookup.get(class_a, set())
        instructors_b = instructor_lookup.get(class_b, set())
        shared_instructors = instructors_a.intersection(instructors_b)

        conflict_types = []

        if shared_rooms:
            conflict_types.append("ROOM")
            room_conflict_count += 1

        if shared_students:
            conflict_types.append("STUDENT")
            student_conflict_count += 1

        if shared_instructors:
            conflict_types.append("INSTRUCTOR")
            instructor_conflict_count += 1

        if not conflict_types:
            continue

        conflict_records.append(
            {
                "class_id_a": class_a,
                "class_id_b": class_b,
                "room_ids_a": ",".join(sorted(a["room_ids"])),
                "room_ids_b": ",".join(sorted(b["room_ids"])),
                "shared_room_ids": ",".join(sorted(shared_rooms)),
                "start_minute_a": a["start_minute"],
                "end_minute_a": a["end_minute"],
                "start_minute_b": b["start_minute"],
                "end_minute_b": b["end_minute"],
                "conflict_types": ",".join(conflict_types),
                "shared_student_count": len(shared_students),
                "shared_instructor_count": len(shared_instructors),
            }
        )

    output_columns = [
        "class_id_a",
        "class_id_b",
        "room_ids_a",
        "room_ids_b",
        "shared_room_ids",
        "start_minute_a",
        "end_minute_a",
        "start_minute_b",
        "end_minute_b",
        "conflict_types",
        "shared_student_count",
        "shared_instructor_count",
    ]

    conflicts = pd.DataFrame(
        conflict_records,
        columns=output_columns,
    )

    summary = {
        "selected_assignment_rows": len(assignments),
        "unique_scheduled_classes": len(schedules),
        "overlapping_class_pairs_checked": overlapping_pairs_checked,
        "unique_class_pairs_with_conflicts": len(conflicts),
        "room_conflict_pairs": room_conflict_count,
        "student_conflict_pairs": student_conflict_count,
        "instructor_conflict_pairs": instructor_conflict_count,
        "classes_with_enrollment_data": len(student_lookup),
        "classes_with_recorded_instructor_data": sum(
            bool(instructors)
            for instructors in instructor_lookup.values()
        ),
        "total_classes_in_metadata": len(classes),
    }

    return conflicts, summary


def main():
    print("=" * 65)
    print("UniSched-AI | UniTime Benchmark Conflict Analysis")
    print("=" * 65)

    assignments, classes, enrollments = load_data()

    print(f"Selected assignment rows: {len(assignments)}")
    print(f"Class metadata rows: {len(classes)}")
    print(f"Student-class enrollment links: {len(enrollments)}")

    conflicts, summary = analyze_conflicts(
        assignments,
        classes,
        enrollments,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORT_DIR / "unitime_conflicts.csv"
    conflicts.to_csv(output_path, index=False)

    print("\n" + "-" * 65)
    print("CONFLICT ANALYSIS SUMMARY")
    print("-" * 65)

    for name, value in summary.items():
        print(f"{name.replace('_', ' ').title()}: {value}")

    print(f"\nDetailed conflict report: {output_path}")

    if conflicts.empty:
        print(
            "\nNo conflicts were detected by the implemented checks. "
            "This does not prove every timetable constraint is satisfied."
        )
    else:
        print("\nConflict types:")
        for conflict_type in ["ROOM", "STUDENT", "INSTRUCTOR"]:
            count = conflicts["conflict_types"].apply(
                lambda value: conflict_type in value.split(",")
            ).sum()

            print(
                f"  {conflict_type}: "
                f"{count} conflicting class pairs"
            )

        print("\nFirst 10 detected conflict pairs:")
        print(conflicts.head(10).to_string(index=False))

    print("\nLimitations:")
    print(
        "- Instructor conflicts can be undercounted when instructor "
        "metadata is missing."
    )
    print(
        "- Date-mask alignment must match the source dataset's encoding."
    )
    print(
        "- This analysis checks selected room, student, and instructor "
        "conflicts only; it does not validate every UniTime constraint."
    )
    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
