
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

ROOMS_FILE = ROOT / "data" / "rooms.csv"
FACULTY_FILE = ROOT / "data" / "faculty_availability.csv"

DAY_MAP = {
    "mon": "Monday",
    "monday": "Monday",
    "tue": "Tuesday",
    "tues": "Tuesday",
    "tuesday": "Tuesday",
    "wed": "Wednesday",
    "wednesday": "Wednesday",
    "thu": "Thursday",
    "thur": "Thursday",
    "thurs": "Thursday",
    "thursday": "Thursday",
    "fri": "Friday",
    "friday": "Friday",
    "sat": "Saturday",
    "saturday": "Saturday",
    "sun": "Sunday",
    "sunday": "Sunday"
}


def normalize_day(day):
    return DAY_MAP.get(str(day).strip().lower())


def time_to_minutes(value):
    try:
        parsed = pd.to_datetime(str(value), format="%H:%M")
        return parsed.hour * 60 + parsed.minute
    except (ValueError, TypeError):
        raise ValueError(f"Invalid time format: {value}. Use HH:MM.")


def load_resource_data():
    if not ROOMS_FILE.exists():
        raise FileNotFoundError("data/rooms.csv was not found.")

    if not FACULTY_FILE.exists():
        raise FileNotFoundError(
            "data/faculty_availability.csv was not found."
        )

    rooms = pd.read_csv(ROOMS_FILE)
    faculty = pd.read_csv(FACULTY_FILE)

    required_room_columns = {
        "room", "capacity", "available_days",
        "availability_start", "availability_end"
    }

    required_faculty_columns = {
        "faculty", "available_days",
        "availability_start", "availability_end"
    }

    if not required_room_columns.issubset(rooms.columns):
        raise ValueError("Room data is missing required columns.")

    if not required_faculty_columns.issubset(faculty.columns):
        raise ValueError("Faculty data is missing required columns.")

    return rooms, faculty


def is_available(day, start_time, end_time,
                 available_days, availability_start,
                 availability_end):

    normalized_day = normalize_day(day)

    if normalized_day is None:
        return False, "Unknown day in timetable."

    days = [
        normalize_day(item)
        for item in str(available_days).split(";")
    ]

    if normalized_day not in days:
        return False, f"Not available on {normalized_day}."

    class_start = time_to_minutes(start_time)
    class_end = time_to_minutes(end_time)
    available_start = time_to_minutes(availability_start)
    available_end = time_to_minutes(availability_end)

    if class_end <= class_start:
        return False, "Invalid class time interval."

    if available_end <= available_start:
        return False, "Invalid resource availability interval."

    if class_start < available_start or class_end > available_end:
        return False, "Class is outside available working hours."

    return True, "Available."


def check_resource_availability(schedule):
    required_columns = {
        "class_id", "day", "start_time",
        "end_time", "room", "faculty"
    }

    if not required_columns.issubset(schedule.columns):
        raise ValueError(
            f"Schedule must contain: {sorted(required_columns)}"
        )

    rooms, faculty_data = load_resource_data()

    results = []

    room_lookup = {
        str(row["room"]): row
        for _, row in rooms.iterrows()
    }

    faculty_lookup = {
        str(row["faculty"]): row
        for _, row in faculty_data.iterrows()
    }

    for _, class_row in schedule.iterrows():

        class_id = class_row["class_id"]
        day = class_row["day"]
        start = class_row["start_time"]
        end = class_row["end_time"]

        # Check room availability
        room_name = str(class_row["room"])
        room = room_lookup.get(room_name)

        if room is None:
            results.append({
                "class_id": class_id,
                "resource_type": "Room",
                "resource": room_name,
                "status": "Unavailable",
                "issue": "Room not found in resource data."
            })
        else:
            available, message = is_available(
                day,
                start,
                end,
                room["available_days"],
                room["availability_start"],
                room["availability_end"]
            )

            results.append({
                "class_id": class_id,
                "resource_type": "Room",
                "resource": room_name,
                "status": "Available" if available else "Unavailable",
                "issue": message
            })

            # Check capacity only when student_count exists
            if "student_count" in schedule.columns:
                student_count = class_row["student_count"]
                capacity = pd.to_numeric(
                    room["capacity"], errors="coerce"
                )

                if pd.notna(student_count) and pd.notna(capacity):
                    if float(student_count) > float(capacity):
                        results.append({
                            "class_id": class_id,
                            "resource_type": "Room Capacity",
                            "resource": room_name,
                            "status": "Unavailable",
                            "issue": (
                                f"{int(student_count)} students exceed "
                                f"room capacity of {int(capacity)}."
                            )
                        })
                    else:
                        results.append({
                            "class_id": class_id,
                            "resource_type": "Room Capacity",
                            "resource": room_name,
                            "status": "Available",
                            "issue": "Room capacity is sufficient."
                        })

        # Check faculty availability
        faculty_name = str(class_row["faculty"])
        faculty = faculty_lookup.get(faculty_name)

        if faculty is None:
            results.append({
                "class_id": class_id,
                "resource_type": "Faculty",
                "resource": faculty_name,
                "status": "Unavailable",
                "issue": "Faculty not found in resource data."
            })
        else:
            available, message = is_available(
                day,
                start,
                end,
                faculty["available_days"],
                faculty["availability_start"],
                faculty["availability_end"]
            )

            results.append({
                "class_id": class_id,
                "resource_type": "Faculty",
                "resource": faculty_name,
                "status": "Available" if available else "Unavailable",
                "issue": message
            })

    return pd.DataFrame(results)


def get_resource_summary(results):
    if results.empty:
        return {
            "total_checks": 0,
            "available": 0,
            "unavailable": 0
        }

    return {
        "total_checks": len(results),
        "available": int((results["status"] == "Available").sum()),
        "unavailable": int(
            (results["status"] == "Unavailable").sum()
        )
    }
