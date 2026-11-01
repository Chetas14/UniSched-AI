
import pandas as pd

from src.conflict_detector import (
    detect_room_conflicts,
    detect_faculty_conflicts,
    detect_student_group_conflicts,
)


def test_detects_overlapping_room_classes():
    schedule = pd.DataFrame([
        {
            "class_id": "C1",
            "course": "Data Structures",
            "day": "Monday",
            "start_time": "09:00",
            "end_time": "10:00",
            "room": "R101",
        },
        {
            "class_id": "C2",
            "course": "Operating Systems",
            "day": "Monday",
            "start_time": "09:30",
            "end_time": "10:30",
            "room": "R101",
        },
    ])

    conflicts = detect_room_conflicts(schedule)

    assert len(conflicts) == 1
    assert conflicts.iloc[0]["overlap_start"] == "09:30"
    assert conflicts.iloc[0]["overlap_end"] == "10:00"


def test_back_to_back_classes_are_not_a_conflict():
    schedule = pd.DataFrame([
        {
            "class_id": "C1",
            "course": "Data Structures",
            "day": "Monday",
            "start_time": "09:00",
            "end_time": "10:00",
            "room": "R101",
        },
        {
            "class_id": "C2",
            "course": "Operating Systems",
            "day": "Monday",
            "start_time": "10:00",
            "end_time": "11:00",
            "room": "R101",
        },
    ])

    conflicts = detect_room_conflicts(schedule)

    assert conflicts.empty


def test_detects_faculty_conflict():
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
            "room": "R102",
            "faculty": "Dr. Sharma",
            "student_group": "CSE-B",
        },
    ])

    conflicts = detect_faculty_conflicts(schedule)

    assert len(conflicts) == 1
    assert conflicts.iloc[0]["resource"] == "Dr. Sharma"


def test_detects_student_group_conflict():
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
            "room": "R102",
            "faculty": "Dr. Kumar",
            "student_group": "CSE-A",
        },
    ])

    conflicts = detect_student_group_conflicts(schedule)

    assert len(conflicts) == 1
    assert conflicts.iloc[0]["resource"] == "CSE-A"
