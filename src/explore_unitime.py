
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/processed")

classes = pd.read_csv(DATA_DIR / "unitime_classes.csv")
candidates = pd.read_csv(
    DATA_DIR / "unitime_candidate_assignments.csv"
)
rooms = pd.read_csv(DATA_DIR / "unitime_rooms.csv")
enrollments = pd.read_csv(
    DATA_DIR / "unitime_student_class_enrollments.csv"
)
constraints = pd.read_csv(
    DATA_DIR / "unitime_group_constraints.csv"
)

print("=" * 60)
print("UniSched-AI: UniTime Benchmark EDA")
print("=" * 60)

print("\n1. DATASET SHAPES")
for name, df in [
    ("Classes", classes),
    ("Candidate assignments", candidates),
    ("Rooms", rooms),
    ("Student-class links", enrollments),
    ("Group constraints", constraints),
]:
    print(f"{name}: {df.shape[0]} rows, {df.shape[1]} columns")

print("\n2. CLASS DATA")
print("Columns:", classes.columns.tolist())
print(classes.head(5).to_string(index=False))

print("\n3. CANDIDATE ASSIGNMENTS")
print("Columns:", candidates.columns.tolist())
print(candidates.head(5).to_string(index=False))

print("\n4. CANDIDATE OPTIONS PER CLASS")
options = candidates.groupby("class_id").size()

print(options.describe().round(2).to_string())
print(f"Classes with candidate options: {options.size}")
print(f"Minimum options: {options.min()}")
print(f"Maximum options: {options.max()}")
print(f"Average options: {options.mean():.2f}")

print("\n5. ROOM AVAILABILITY IN CANDIDATES")
print(
    candidates["room_id"]
    .value_counts()
    .head(10)
    .to_string()
)

print("\n6. STUDENT ENROLLMENT LINKS PER CLASS")
enrollment_counts = enrollments.groupby("class_id").size()

print(enrollment_counts.describe().round(2).to_string())

print("\n7. GROUP CONSTRAINT TYPES")
if "constraint_type" in constraints.columns:
    print(constraints["constraint_type"].value_counts().to_string())
else:
    print("Available columns:", constraints.columns.tolist())

print("\n8. MISSING VALUES")
for name, df in [
    ("Classes", classes),
    ("Candidates", candidates),
    ("Rooms", rooms),
    ("Enrollments", enrollments),
    ("Constraints", constraints),
]:
    missing = int(df.isna().sum().sum())
    print(f"{name}: {missing} missing cells")

print("\nEDA completed.")
