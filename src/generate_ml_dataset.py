
from pathlib import Path
import numpy as np
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------

RANDOM_SEED = 42
NUMBER_OF_SCHEDULES = 20
NUMBER_OF_CLASSES = 500
NUMBER_OF_CONFLICT_PAIRS = 120

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "ml_timetable_dataset.csv"

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

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

START_TIMES = [f"{hour:02d}:00" for hour in range(8, 17)]
CLASS_DURATION = 60


# -----------------------------
# Time conversion
# -----------------------------

def to_minutes(time_string):
    hour, minute = map(int, time_string.split(":"))
    return hour * 60 + minute


def from_minutes(minutes):
    hour = minutes // 60
    minute = minutes % 60
    return f"{hour:02d}:{minute:02d}"


# -----------------------------
# Generate one timetable
# -----------------------------

def generate_base_schedule(rng, schedule_id):

    rows = []

    for i in range(NUMBER_OF_CLASSES):

        start_time = rng.choice(START_TIMES)
        start_minutes = to_minutes(start_time)

        end_time = from_minutes(
            start_minutes + CLASS_DURATION
        )

        rows.append({
            "schedule_id": schedule_id,
            "class_id": f"{schedule_id}_C{i + 1:04d}",
            "course": rng.choice(COURSES),
            "day": DAYS[i % len(DAYS)],
            "start_time": start_time,
            "end_time": end_time,
            "room": f"{schedule_id}_ROOM_{i + 1:04d}",
            "faculty": f"{schedule_id}_FACULTY_{i + 1:04d}",
            "student_group": f"{schedule_id}_GROUP_{i + 1:04d}",
            "duration_minutes": CLASS_DURATION,
        })

    return pd.DataFrame(rows)


# -----------------------------
# Introduce controlled conflicts
# -----------------------------

def introduce_conflicts(df, rng):

    pair_starts = list(
        range(0, NUMBER_OF_CONFLICT_PAIRS * 2, 2)
    )

    conflict_types = (
        ["room"] * 40
        + ["faculty"] * 40
        + ["student_group"] * 40
    )

    rng.shuffle(conflict_types)

    for start, conflict_type in zip(
        pair_starts, conflict_types
    ):

        first = start
        second = start + 1

        # Same day and time
        day = rng.choice(DAYS)
        start_time = rng.choice(START_TIMES)

        start_minutes = to_minutes(start_time)
        end_time = from_minutes(
            start_minutes + CLASS_DURATION
        )

        for index in [first, second]:

            df.at[index, "day"] = day
            df.at[index, "start_time"] = start_time
            df.at[index, "end_time"] = end_time

        # Share exactly one resource
        df.at[second, conflict_type] = df.at[
            first, conflict_type
        ]

    return df


# -----------------------------
# Calculate conflicts
# -----------------------------

def calculate_conflicts(df):

    conflict_counts = [0] * len(df)

    for i in range(len(df)):

        for j in range(i + 1, len(df)):

            if df.iloc[i]["day"] != df.iloc[j]["day"]:
                continue

            start_i = to_minutes(df.iloc[i]["start_time"])
            end_i = to_minutes(df.iloc[i]["end_time"])

            start_j = to_minutes(df.iloc[j]["start_time"])
            end_j = to_minutes(df.iloc[j]["end_time"])

            overlaps = (
                start_i < end_j and start_j < end_i
            )

            if not overlaps:
                continue

            shared_resource = (
                df.iloc[i]["room"] == df.iloc[j]["room"]
                or df.iloc[i]["faculty"] == df.iloc[j]["faculty"]
                or df.iloc[i]["student_group"]
                == df.iloc[j]["student_group"]
            )

            if shared_resource:
                conflict_counts[i] += 1
                conflict_counts[j] += 1

    df["conflict_count"] = conflict_counts

    df["has_conflict"] = (
        df["conflict_count"] > 0
    ).astype(int)

    return df


# -----------------------------
# Generate complete ML dataset
# -----------------------------

def generate_ml_dataset():

    all_schedules = []

    for number in range(1, NUMBER_OF_SCHEDULES + 1):

        schedule_id = f"S{number:03d}"

        print(f"Generating {schedule_id}...")

        rng = np.random.default_rng(
            RANDOM_SEED + number
        )

        df = generate_base_schedule(
            rng, schedule_id
        )

        df = introduce_conflicts(df, rng)

        # Calculate conflicts within this timetable only
        df = calculate_conflicts(df)

        df["data_source"] = "synthetic"

        all_schedules.append(df)

    final_df = pd.concat(
        all_schedules,
        ignore_index=True
    )

    return final_df


# -----------------------------
# Main
# -----------------------------

def main():

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = generate_ml_dataset()

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nDataset generated successfully!")
    print("Output:", OUTPUT_FILE)
    print("Total records:", len(df))
    print("Total schedules:", df["schedule_id"].nunique())

    print("\nClass distribution:")
    print(df["has_conflict"].value_counts())

    print("\nRecords per schedule:")
    print(df.groupby("schedule_id").size().head())

    print("\nDataset saved successfully!")


if __name__ == "__main__":
    main()
