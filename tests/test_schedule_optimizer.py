
import pandas as pd

from src.conflict_detector import (
    detect_room_conflicts,
    detect_faculty_conflicts,
    detect_student_group_conflicts,
)
from src.schedule_optimizer import optimize_schedule


def test_optimizer_resolves_all_conflicts():
    schedule = pd.DataFrame([
        {
            "class_id": "C1",
            "course": "Data Structures",
            "day": "Monday",
            "start_time": "09:00",
            "end_time": "10:00",
            "room": "R101",
            "faculty": "Dr. Sharma",
            "student_group": "CSE-A",
        },
        {
            "class_id": "C2",
            "course": "Operating Systems",
            "day": "Monday",
            "start_time": "09:30",
            "end_time": "10:30",
            "room": "R101",
            "faculty": "Dr. Kumar",
            "student_group": "CSE-B",
        },
    ])

    result = optimize_schedule(
        schedule,
        candidate_start_times=[
            "09:00",
            "09:30",
            "10:00",
            "10:30",
            "11:00",
        ],
    )

    assert result["schedule"] is not None

    optimized = result["schedule"]

    assert detect_room_conflicts(optimized).empty
    assert detect_faculty_conflicts(optimized).empty
    assert detect_student_group_conflicts(optimized).empty


def test_optimizer_preserves_class_durations():
    schedule = pd.DataFrame([
        {
            "class_id": "C1",
            "course": "Data Structures",
            "day": "Monday",
            "start_time": "09:00",
            "end_time": "10:00",
            "room": "R101",
            "faculty": "Dr. Sharma",
            "student_group": "CSE-A",
        },
        {
            "class_id": "C2",
            "course": "Operating Systems",
            "day": "Monday",
            "start_time": "09:30",
            "end_time": "11:00",
            "room": "R102",
            "faculty": "Dr. Kumar",
            "student_group": "CSE-B",
        },
    ])

    result = optimize_schedule(
        schedule,
        candidate_start_times=[
            "09:00",
            "09:30",
            "10:00",
            "10:30",
            "11:00",
            "11:30",
        ],
    )

    assert result["schedule"] is not None

    optimized = result["schedule"]

    original_durations = (
        pd.to_datetime(schedule["end_time"], format="%H:%M")
        - pd.to_datetime(schedule["start_time"], format="%H:%M")
    )

    optimized_durations = (
        pd.to_datetime(optimized["end_time"], format="%H:%M")
        - pd.to_datetime(optimized["start_time"], format="%H:%M")
    )

    assert list(original_durations) == list(optimized_durations)
