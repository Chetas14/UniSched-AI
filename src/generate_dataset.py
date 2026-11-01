
from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
NUMBER_OF_CLASSES = 500

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "timetable_dataset.csv"

DAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
]

COURSES = [
    "Data Structures",
    "Operating Systems",
    "Database Systems",
    "Computer Networks",
    "Software Engineering",
    "Machine Learning",
    "Artificial Intelligence",
    "Computer Architecture",
    "Web Development",
    "Cloud Computing",
]

START_TIMES = [
    f"{hour:02d}:00"
    for hour in range(8, 17)
]

CLASS_DURATION = 60


def to_minutes(time_string):
    """Convert HH:MM into minutes after midnight."""

    hour, minute = map(int, time_string.split(":"))
    return hour * 60 + minute


def from_minutes(minutes):
    """Convert minutes after midnight into HH:MM."""

    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def generate_base_schedule(rng):
    """Generate classes with initially unique resources."""

    rows = []

    for i in range(NUMBER_OF_CLASSES):
        start_time = START_TIMES[
            (i // len(DAYS)) % len(START_TIMES)
        ]

        start_minutes = to_minutes(start_time)

        rows.append({
            "class_id": f"C{i + 1:04d}",
            "course": str(rng.choice(COURSES)),
            "day": DAYS[i % len(DAYS)],
            "start_time": start_time,
            "end_time": from_minutes(
                start_minutes + CLASS_DURATION
            ),
            "room": f"ROOM_{i + 1:04d}",
            "faculty": f"FACULTY_{i + 1:04d}",
            "student_group": f"GROUP_{i + 1:04d}",
            "duration_minutes": CLASS_DURATION,
        })

    return pd.DataFrame(rows)


def introduce_conflicts(df, rng):
    """
    Introduce room, faculty, and student-group conflicts.

    Every injected conflict uses a pair of classes with the
    same day and time and one shared resource.
    """

    data = df.copy()

    # Use disjoint pairs to avoid reusing a class across
    # multiple injected scenarios.
    pair_starts = list(range(0, 240, 2))
    rng.shuffle(pair_starts)

    conflict_types = (
        ["room"] * 40
        + ["faculty"] * 40
        + ["student_group"] * 40
    )

    rng.shuffle(conflict_types)

    for pair_start, resource_type in zip(
        pair_starts,
        conflict_types,
    ):
        first_idx = pair_start
        second_idx = pair_start + 1

        # Put both classes on the same day and time.
        shared_day = DAYS[
            int(rng.integers(0, len(DAYS)))
        ]

        shared_start = START_TIMES[
            int(rng.integers(0, len(START_TIMES)))
        ]

        start_minutes = to_minutes(shared_start)
        shared_end = from_minutes(
            start_minutes + CLASS_DURATION
        )

        data.at[first_idx, "day"] = shared_day
        data.at[second_idx, "day"] = shared_day

        data.at[first_idx, "start_time"] = shared_start
        data.at[second_idx, "start_time"] = shared_start

        data.at[first_idx, "end_time"] = shared_end
        data.at[second_idx, "end_time"] = shared_end

        # Make exactly one resource shared by the pair.
        shared_resource = data.at[first_idx, resource_type]

        data.at[second_idx, resource_type] = shared_resource

    return data


def calculate_conflicts(df):
    """Calculate labels using actual timetable overlaps."""

    data = df.copy()

    starts = data["start_time"].map(to_minutes).to_numpy()
    ends = data["end_time"].map(to_minutes).to_numpy()

    conflict_count = np.zeros(len(data), dtype=int)
    resources = ["room", "faculty", "student_group"]

    for i in range(len(data)):
        for j in range(i + 1, len(data)):

            if data.at[i, "day"] != data.at[j, "day"]:
                continue

            overlaps = (
                starts[i] < ends[j]
                and starts[j] < ends[i]
            )

            if not overlaps:
                continue

            shared_resources = sum(
                data.at[i, resource] == data.at[j, resource]
                for resource in resources
            )

            if shared_resources:
                conflict_count[i] += 1
                conflict_count[j] += 1

    data["conflict_count"] = conflict_count
    data["has_conflict"] = (
        data["conflict_count"] > 0
    ).astype(int)

    return data


def generate_dataset():
    """Generate a reproducible synthetic timetable dataset."""

    rng = np.random.default_rng(RANDOM_SEED)

    base = generate_base_schedule(rng)
    modified = introduce_conflicts(base, rng)
    result = calculate_conflicts(modified)

    result["data_source"] = "synthetic"

    return result


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = generate_dataset()
    df.to_csv(OUTPUT_FILE, index=False)

    print("Synthetic timetable dataset generated.")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Total classes: {len(df)}")

    print("\nTarget label distribution:")
    print(
        df["has_conflict"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
        .sort_index()
    )

    print("\nConflict counts by resource type:")

    for resource in ["room", "faculty", "student_group"]:
        pairs = 0

        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                if df.at[i, "day"] != df.at[j, "day"]:
                    continue

                start_i = to_minutes(df.at[i, "start_time"])
                end_i = to_minutes(df.at[i, "end_time"])
                start_j = to_minutes(df.at[j, "start_time"])
                end_j = to_minutes(df.at[j, "end_time"])

                overlaps = start_i < end_j and start_j < end_i

                if (
                    overlaps
                    and df.at[i, resource] == df.at[j, resource]
                ):
                    pairs += 1

        print(f"{resource}: {pairs} conflict pairs")

    print("\nFirst five rows:")
    print(df.head().to_string(index=False))

    if df["has_conflict"].nunique() < 2:
        raise RuntimeError(
            "Dataset contains only one target class."
        )


if __name__ == "__main__":
    main()
