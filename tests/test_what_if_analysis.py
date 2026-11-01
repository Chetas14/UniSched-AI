
import pandas as pd

from src.what_if_analysis import (
    modify_class_scenario,
    count_schedule_conflicts,
    run_what_if_scenario,
)


def create_sample_schedule():
    return pd.DataFrame([
        {
            "class_id": "C1",
            "course": "Python",
            "day": "Monday",
            "start_time": "09:00",
            "end_time": "10:00",
            "room": "R1",
            "faculty": "F1",
            "student_group": "G1",
        },
        {
            "class_id": "C2",
            "course": "Java",
            "day": "Monday",
            "start_time": "09:30",
            "end_time": "10:30",
            "room": "R1",
            "faculty": "F2",
            "student_group": "G2",
        },
    ])


def test_scenario_preserves_original_schedule():
    schedule = create_sample_schedule()

    scenario = modify_class_scenario(
        schedule,
        class_id="C1",
        new_start_time="11:00",
    )

    assert schedule.loc[0, "start_time"] == "09:00"
    assert scenario.loc[0, "start_time"] == "11:00"
    assert scenario.loc[0, "end_time"] == "12:00"


def test_scenario_detects_conflict_reduction():
    schedule = create_sample_schedule()

    before = count_schedule_conflicts(schedule)

    scenario = modify_class_scenario(
        schedule,
        class_id="C1",
        new_start_time="11:00",
    )

    after = count_schedule_conflicts(scenario)

    assert before["total"] == 1
    assert after["total"] == 0


def test_unknown_class_id_raises_error():
    schedule = create_sample_schedule()

    try:
        modify_class_scenario(
            schedule,
            class_id="C999",
            new_start_time="11:00",
        )
        assert False, "Expected ValueError"
    except ValueError:
        pass
