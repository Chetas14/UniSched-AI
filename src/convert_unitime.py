
"""
UniSched-AI: UniTime Dataset Converter

Reads the public UniTime XML benchmark from a ZIP archive and exports:
1. Class metadata
2. Candidate room/time assignments
3. Selected timetable assignments
4. Room metadata
5. Student-class enrollment links
6. Group constraints

Candidate assignments are possible placements, not actual scheduled classes.
Only XML options marked solution="true" are treated as selected assignments.
Classes requiring multiple rooms retain one row per selected room.
"""

from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import pandas as pd


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ZIP_PATH = PROJECT_ROOT / "data" / "raw" / "pu-fal07-cs.zip"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# XML LOADING
# --------------------------------------------------

def read_xml_from_zip(zip_path=ZIP_PATH):
    """Read the UniTime XML file from the downloaded ZIP archive."""

    if not zip_path.exists():
        raise FileNotFoundError(
            f"Dataset ZIP file not found: {zip_path}\n"
            "Check that pu-fal07-cs.zip is inside data/raw/."
        )

    with ZipFile(zip_path, "r") as archive:
        xml_files = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".xml")
            and not name.startswith("__MACOSX/")
        ]

        if not xml_files:
            raise FileNotFoundError(
                "No XML file was found inside the ZIP archive."
            )

        # Prefer an XML file at the archive root.
        root_level_files = [
            name for name in xml_files if "/" not in name
        ]
        selected_file = (
            root_level_files[0] if root_level_files else xml_files[0]
        )

        xml_content = archive.read(selected_file)

    root = ET.fromstring(xml_content)

    print(f"Read XML file: {selected_file}")

    return root


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def get_direct_children(root, parent_name, child_name):
    """Return direct XML children without matching nested elements."""

    parent = root.find(parent_name)

    if parent is None:
        return []

    return parent.findall(child_name)


def safe_int(value, default=None):
    """Convert an XML attribute to an integer safely."""

    if value is None or value == "":
        return default

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def is_selected(element):
    """Check whether an XML option is marked as selected."""

    return element.get("solution", "").strip().lower() == "true"


# --------------------------------------------------
# CLASS METADATA
# --------------------------------------------------

def get_class_data(root):
    """Extract one metadata row per actual timetable class."""

    rows = []
    class_elements = get_direct_children(root, "classes", "class")

    for class_element in class_elements:
        rows.append({
            "class_id": class_element.get("id"),
            "offering_id": class_element.get("offering"),
            "config_id": class_element.get("config"),
            "subpart_id": class_element.get("subpart"),
            "parent_id": class_element.get("parent"),
            "class_limit": safe_int(
                class_element.get("classLimit")
            ),
            "committed": class_element.get("committed"),
            "scheduler_id": class_element.get("scheduler"),
            "dates_mask": class_element.get("dates"),
            "instructor_ids": ",".join(
                instructor.get("id", "")
                for instructor in class_element.findall("instructor")
            ),
            "candidate_room_count": len(
                class_element.findall("room")
            ),
            "candidate_time_count": len(
                class_element.findall("time")
            ),
        })

    columns = [
        "class_id",
        "offering_id",
        "config_id",
        "subpart_id",
        "parent_id",
        "class_limit",
        "committed",
        "scheduler_id",
        "dates_mask",
        "instructor_ids",
        "candidate_room_count",
        "candidate_time_count",
    ]

    return pd.DataFrame(rows, columns=columns)


# --------------------------------------------------
# CANDIDATE ASSIGNMENTS
# --------------------------------------------------

def get_candidate_assignments(root):
    """
    Extract possible room/time combinations for each class.

    Candidate combinations are not automatically treated as scheduled.
    """

    rows = []
    class_elements = get_direct_children(root, "classes", "class")

    for class_element in class_elements:
        class_id = class_element.get("id")
        dates_mask = class_element.get("dates")

        time_options = class_element.findall("time")
        room_options = class_element.findall("room")

        for time_index, time_element in enumerate(time_options):
            start_slot = safe_int(time_element.get("start"))
            duration_slots = safe_int(time_element.get("length"))

            if (
                start_slot is None
                or duration_slots is None
                or duration_slots <= 0
            ):
                continue

            start_minute = start_slot * 5
            duration_minutes = duration_slots * 5
            end_minute = start_minute + duration_minutes

            for room_index, room_element in enumerate(room_options):
                selected_time = is_selected(time_element)
                selected_room = is_selected(room_element)

                if selected_time and selected_room:
                    assignment_status = "selected"
                else:
                    assignment_status = "candidate_not_final"

                rows.append({
                    "class_id": class_id,
                    "time_option_id": time_index,
                    "room_option_id": room_index,
                    "room_id": room_element.get("id"),
                    "days_mask": time_element.get("days"),
                    "dates_mask": dates_mask,
                    "start_slot": start_slot,
                    "duration_slots": duration_slots,
                    "start_minute": start_minute,
                    "end_minute": end_minute,
                    "duration_minutes": duration_minutes,
                    "time_preference": time_element.get("pref"),
                    "room_preference": room_element.get("pref"),
                    "time_solution": selected_time,
                    "room_solution": selected_room,
                    "assignment_status": assignment_status,
                })

    columns = [
        "class_id",
        "time_option_id",
        "room_option_id",
        "room_id",
        "days_mask",
        "dates_mask",
        "start_slot",
        "duration_slots",
        "start_minute",
        "end_minute",
        "duration_minutes",
        "time_preference",
        "room_preference",
        "time_solution",
        "room_solution",
        "assignment_status",
    ]

    return pd.DataFrame(rows, columns=columns)


# --------------------------------------------------
# SELECTED TIMETABLE
# --------------------------------------------------

def get_selected_assignments(root):
    """
    Extract selected timetable assignments.

    A class must have exactly one selected time and at least one selected
    room. Each selected room is preserved as a separate row.

    Classes with missing or ambiguous selected times are excluded rather
    than being assigned guessed values.
    """

    rows = []
    class_elements = get_direct_children(root, "classes", "class")

    for class_element in class_elements:
        class_id = class_element.get("id")

        selected_times = [
            element
            for element in class_element.findall("time")
            if is_selected(element)
        ]

        selected_rooms = [
            element
            for element in class_element.findall("room")
            if is_selected(element)
        ]

        # Do not guess when the selected time is missing or ambiguous.
        if len(selected_times) != 1 or not selected_rooms:
            continue

        time_element = selected_times[0]

        start_slot = safe_int(time_element.get("start"))
        duration_slots = safe_int(time_element.get("length"))

        if (
            start_slot is None
            or duration_slots is None
            or duration_slots <= 0
        ):
            continue

        start_minute = start_slot * 5
        duration_minutes = duration_slots * 5
        end_minute = start_minute + duration_minutes

        # Preserve every selected room for multi-room classes.
        for room_element in selected_rooms:
            room_id = room_element.get("id")

            if room_id is None:
                continue

            rows.append({
                "class_id": class_id,
                "room_id": room_id,
                "days_mask": time_element.get("days"),
                "dates_mask": class_element.get("dates"),
                "start_slot": start_slot,
                "duration_slots": duration_slots,
                "start_minute": start_minute,
                "end_minute": end_minute,
                "duration_minutes": duration_minutes,
                "time_preference": time_element.get("pref"),
                "room_preference": room_element.get("pref"),
                "assignment_status": "selected",
            })

    columns = [
        "class_id",
        "room_id",
        "days_mask",
        "dates_mask",
        "start_slot",
        "duration_slots",
        "start_minute",
        "end_minute",
        "duration_minutes",
        "time_preference",
        "room_preference",
        "assignment_status",
    ]

    return pd.DataFrame(rows, columns=columns)


# --------------------------------------------------
# ROOM METADATA
# --------------------------------------------------

def get_room_data(root):
    """Extract room IDs and available metadata."""

    rows = []
    room_elements = get_direct_children(root, "rooms", "room")

    for room_element in room_elements:
        rows.append({
            "room_id": room_element.get("id"),
            "capacity": safe_int(room_element.get("capacity")),
            "location": room_element.get("location"),
            "constraint": room_element.get("constraint"),
        })

    columns = [
        "room_id",
        "capacity",
        "location",
        "constraint",
    ]

    return pd.DataFrame(rows, columns=columns)


# --------------------------------------------------
# STUDENT-CLASS ENROLLMENT LINKS
# --------------------------------------------------

def get_student_class_links(root):
    """Extract explicit links between students and classes."""

    rows = []
    student_elements = get_direct_children(
        root, "students", "student"
    )

    for student_element in student_elements:
        student_id = student_element.get("id")

        for class_element in student_element.iter("class"):
            class_id = class_element.get("id")

            if student_id is not None and class_id is not None:
                rows.append({
                    "student_id": student_id,
                    "class_id": class_id,
                })

    columns = ["student_id", "class_id"]
    dataframe = pd.DataFrame(rows, columns=columns)

    if not dataframe.empty:
        dataframe = dataframe.drop_duplicates()

    return dataframe


# --------------------------------------------------
# GROUP CONSTRAINTS
# --------------------------------------------------

def get_group_constraints(root):
    """Extract group constraints and their referenced classes."""

    rows = []
    constraint_parent = root.find("groupConstraints")

    if constraint_parent is None:
        return pd.DataFrame(
            columns=[
                "constraint_id",
                "constraint_type",
                "preference",
                "class_id",
                "reference_count",
            ]
        )

    for constraint_element in constraint_parent.findall("constraint"):
        constraint_id = constraint_element.get("id")
        constraint_type = (
            constraint_element.get("type")
            or constraint_element.get("constraintType")
        )
        preference = constraint_element.get("pref")

        referenced_classes = [
            element.get("id")
            for element in constraint_element.iter("class")
            if element.get("id") is not None
        ]

        for class_id in referenced_classes:
            rows.append({
                "constraint_id": constraint_id,
                "constraint_type": constraint_type,
                "preference": preference,
                "class_id": class_id,
                "reference_count": len(referenced_classes),
            })

    columns = [
        "constraint_id",
        "constraint_type",
        "preference",
        "class_id",
        "reference_count",
    ]

    return pd.DataFrame(rows, columns=columns)


# --------------------------------------------------
# CSV EXPORT
# --------------------------------------------------

def save_dataframe(dataframe, filename):
    """Save a dataframe to the processed-data folder."""

    output_path = PROCESSED_DIR / filename
    dataframe.to_csv(output_path, index=False)

    print(
        f"Saved {filename}: "
        f"{len(dataframe):,} rows, "
        f"{len(dataframe.columns)} columns"
    )


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def main():
    """Run the complete UniTime XML conversion pipeline."""

    print("=" * 60)
    print("UniSched-AI | UniTime Dataset Converter")
    print("=" * 60)

    root = read_xml_from_zip()

    classes_df = get_class_data(root)
    candidates_df = get_candidate_assignments(root)
    selected_df = get_selected_assignments(root)
    rooms_df = get_room_data(root)
    student_links_df = get_student_class_links(root)
    constraints_df = get_group_constraints(root)

    # Save all datasets separately.
    save_dataframe(classes_df, "unitime_classes.csv")
    save_dataframe(
        candidates_df,
        "unitime_candidate_assignments.csv",
    )
    save_dataframe(
        selected_df,
        "unitime_selected_assignments.csv",
    )
    save_dataframe(rooms_df, "unitime_rooms.csv")
    save_dataframe(
        student_links_df,
        "unitime_student_class_enrollments.csv",
    )
    save_dataframe(
        constraints_df,
        "unitime_group_constraints.csv",
    )

    # Report both unique classes and class-room assignment rows.
    total_classes = len(classes_df)
    assigned_class_ids = set(
        selected_df["class_id"].dropna().astype(str)
    )
    all_class_ids = set(
        classes_df["class_id"].dropna().astype(str)
    )
    unassigned_classes = len(all_class_ids - assigned_class_ids)

    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    print(f"Total classes: {total_classes:,}")
    print(f"Classes with selected assignments: {len(assigned_class_ids):,}")
    print(f"Selected class-room assignment rows: {len(selected_df):,}")
    print(f"Classes without selected assignments: {unassigned_classes:,}")
    print(f"Candidate room/time combinations: {len(candidates_df):,}")
    print(f"Rooms: {len(rooms_df):,}")
    print(f"Student-class links: {len(student_links_df):,}")
    print(f"Group-constraint references: {len(constraints_df):,}")

    print("\nConversion completed.")
    print(f"Output directory: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
