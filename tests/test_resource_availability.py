
import pandas as pd
import pytest

from src.resource_availability import (
    is_available,
    check_resource_availability,
    get_resource_summary
)


def test_resource_available_during_working_hours():
    result, message = is_available(
        "Monday", "09:00", "10:00",
        "Monday;Tuesday;Wednesday;Thursday;Friday",
        "08:00", "18:00"
    )

    assert result is True
    assert message == "Available."


def test_resource_unavailable_on_day():
    result, message = is_available(
        "Saturday", "09:00", "10:00",
        "Monday;Tuesday;Wednesday;Thursday;Friday",
        "08:00", "18:00"
    )

    assert result is False
    assert "Not available" in message


def test_resource_unavailable_outside_working_hours():
    result, message = is_available(
        "Monday", "07:00", "09:00",
        "Monday;Tuesday;Wednesday;Thursday;Friday",
        "08:00", "18:00"
    )

    assert result is False
    assert "outside available working hours" in message


def test_invalid_time_interval():
    result, message = is_available(
        "Monday", "10:00", "09:00",
        "Monday;Tuesday;Wednesday;Thursday;Friday",
        "08:00", "18:00"
    )

    assert result is False
    assert "Invalid class time interval" in message


def test_missing_resource_is_reported(tmp_path, monkeypatch):
    import src.resource_availability as resource_module

    rooms_file = tmp_path / "rooms.csv"
    faculty_file = tmp_path / "faculty.csv"

    pd.DataFrame([{
        "room": "ROOM_1",
        "capacity": 60,
        "available_days": "Monday;Tuesday",
        "availability_start": "08:00",
        "availability_end": "18:00"
    }]).to_csv(rooms_file, index=False)

    pd.DataFrame([{
        "faculty": "FACULTY_1",
        "available_days": "Monday;Tuesday",
        "availability_start": "08:00",
        "availability_end": "18:00"
    }]).to_csv(faculty_file, index=False)

    monkeypatch.setattr(resource_module, "ROOMS_FILE", rooms_file)
    monkeypatch.setattr(resource_module, "FACULTY_FILE", faculty_file)

    schedule = pd.DataFrame([{
        "class_id": "C001",
        "day": "Monday",
        "start_time": "09:00",
        "end_time": "10:00",
        "room": "UNKNOWN_ROOM",
        "faculty": "UNKNOWN_FACULTY"
    }])

    result = check_resource_availability(schedule)

    assert len(result) == 2
    assert (result["status"] == "Unavailable").all()


def test_resource_summary():
    results = pd.DataFrame([
        {"status": "Available"},
        {"status": "Available"},
        {"status": "Unavailable"}
    ])

    summary = get_resource_summary(results)

    assert summary["total_checks"] == 3
    assert summary["available"] == 2
    assert summary["unavailable"] == 1
