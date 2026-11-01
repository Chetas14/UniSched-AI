
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_chart(filename):
    """Save the current chart and close its figure."""

    path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    classes = pd.read_csv(
        DATA_DIR / "unitime_classes.csv"
    )

    candidates = pd.read_csv(
        DATA_DIR / "unitime_candidate_assignments.csv"
    )

    # Chart 1: Candidate assignments per class.
    candidate_counts = (
        candidates.groupby("class_id")
        .size()
        .reindex(classes["class_id"], fill_value=0)
    )

    plt.figure(figsize=(9, 5))
    plt.hist(candidate_counts, bins=25, edgecolor="black")
    plt.title("Candidate Assignments per Class")
    plt.xlabel("Number of Candidate Assignments")
    plt.ylabel("Number of Classes")
    save_chart("candidate_assignments_per_class.png")

    # Chart 2: Candidate room options per class.
    plt.figure(figsize=(8, 5))
    plt.hist(
        classes["candidate_room_count"],
        bins=range(
            int(classes["candidate_room_count"].min()),
            int(classes["candidate_room_count"].max()) + 2,
        ),
        align="left",
        edgecolor="black",
    )
    plt.title("Candidate Room Options per Class")
    plt.xlabel("Number of Candidate Room Options")
    plt.ylabel("Number of Classes")
    save_chart("candidate_room_options.png")

    # Chart 3: Candidate time options per class.
    plt.figure(figsize=(8, 5))
    plt.hist(
        classes["candidate_time_count"],
        bins=20,
        edgecolor="black",
    )
    plt.title("Candidate Time Options per Class")
    plt.xlabel("Number of Candidate Time Options")
    plt.ylabel("Number of Classes")
    save_chart("candidate_time_options.png")

    # Chart 4: Missing values in selected metadata fields.
    metadata_columns = [
        "offering_id",
        "config_id",
        "parent_id",
        "class_limit",
        "scheduler_id",
        "instructor_ids",
    ]

    missing_counts = (
        classes[metadata_columns]
        .isna()
        .sum()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 5))
    missing_counts.plot(kind="bar", edgecolor="black")
    plt.title("Missing Values in Class Metadata")
    plt.xlabel("Metadata Field")
    plt.ylabel("Missing Values")
    plt.xticks(rotation=35, ha="right")
    save_chart("class_metadata_missing_values.png")

    print("\nEDA charts created successfully.")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
