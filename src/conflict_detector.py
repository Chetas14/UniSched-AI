
import pandas as pd


def calculate_severity(overlap_minutes):
    """Assign a severity level based on overlap duration."""

    if overlap_minutes <= 15:
        return "Low"
    elif overlap_minutes <= 30:
        return "Medium"
    else:
        return "High"


def _validate_schedule(schedule, required_columns):
    """Validate the timetable columns and time intervals."""

    missing = required_columns - set(schedule.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    data = schedule.copy()

    data["start_time"] = pd.to_datetime(
        data["start_time"],
        format="%H:%M",
        errors="raise",
    )

    data["end_time"] = pd.to_datetime(
        data["end_time"],
        format="%H:%M",
        errors="raise",
    )

    if (data["end_time"] <= data["start_time"]).any():
        raise ValueError(
            "Each class must end after it starts."
        )

    return data


def _detect_conflicts(schedule, resource_column, conflict_type):
    """Find overlapping classes that share the same resource."""

    required_columns = {
        "class_id",
        "course",
        "day",
        "start_time",
        "end_time",
        resource_column,
    }

    data = _validate_schedule(
        schedule,
        required_columns,
    )

    conflicts = []

    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            first = data.iloc[i]
            second = data.iloc[j]

            same_resource = (
                first["day"] == second["day"]
                and first[resource_column]
                == second[resource_column]
            )

            overlaps = (
                first["start_time"] < second["end_time"]
                and second["start_time"] < first["end_time"]
            )

            if same_resource and overlaps:

                overlap_start = max(
                    first["start_time"],
                    second["start_time"],
                )

                overlap_end = min(
                    first["end_time"],
                    second["end_time"],
                )

                overlap_minutes = int(
                    (
                        overlap_end - overlap_start
                    ).total_seconds() / 60
                )

                conflict = {
                    "day": first["day"],
                    "resource": first[resource_column],
                    "class_1": first["course"],
                    "class_2": second["course"],
                    "overlap_start": overlap_start.strftime(
                        "%H:%M"
                    ),
                    "overlap_end": overlap_end.strftime(
                        "%H:%M"
                    ),
                    "overlap_minutes": overlap_minutes,
                    "severity": calculate_severity(
                        overlap_minutes
                    ),
                }

                if conflict_type == "Room Conflict":
                    conflict["room"] = first["room"]
                else:
                    conflict["conflict_type"] = conflict_type

                conflicts.append(conflict)

    if conflict_type == "Room Conflict":
        columns = [
            "day",
            "room",
            "class_1",
            "class_2",
            "overlap_start",
            "overlap_end",
            "overlap_minutes",
            "severity",
        ]
    else:
        columns = [
            "conflict_type",
            "day",
            "resource",
            "class_1",
            "class_2",
            "overlap_start",
            "overlap_end",
            "overlap_minutes",
            "severity",
        ]

    return pd.DataFrame(
        conflicts,
        columns=columns,
    )


def detect_room_conflicts(schedule):
    """Detect overlapping classes assigned to the same room."""

    if "room" not in schedule.columns:
        raise ValueError(
            "Missing required columns: ['room']"
        )

    return _detect_conflicts(
        schedule,
        resource_column="room",
        conflict_type="Room Conflict",
    )


def detect_faculty_conflicts(schedule):
    """Detect faculty members assigned to overlapping classes."""

    return _detect_conflicts(
        schedule,
        resource_column="faculty",
        conflict_type="Faculty Conflict",
    )


def detect_student_group_conflicts(schedule):
    """Detect student groups assigned to overlapping classes."""

    return _detect_conflicts(
        schedule,
        resource_column="student_group",
        conflict_type="Student Group Conflict",
    )
