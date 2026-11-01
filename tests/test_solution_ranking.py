import pandas as pd
import pytest

from src.solution_ranking import (
    calculate_conflicts,
    calculate_total_time_shift,
    count_rescheduled_classes,
    evaluate_schedule,
    rank_solutions,
    recommend_best_solution,
)


@pytest.fixture
def original_schedule():
    return pd.DataFrame([
        {
            "class_id": 1,
            "course": "Java",
            "day": "Monday",
            "start_time": "09:00",
            "end_time": "10:00",
            "room": "R1",
            "faculty": "F1",
            "student_group": "G1",
        },
        {
            "class_id": 2,
            "course": "Python",
            "day": "Monday",
            "start_time": "09:30",
            "end_time": "10:30",
            "room": "R1",
            "faculty": "F2",
            "student_group": "G2",
        },
    ])


@pytest.fixture
def conflict_free_schedule():
    return pd.DataFrame([
        {
            "class_id": 1,
            "course": "Java",
            "day": "Monday",
            "start_time": "09:00",
            "end_time": "10:00",
            "room": "R1",
            "faculty": "F1",
            "student_group": "G1",
        },
        {
            "class_id": 2,
            "course": "Python",
            "day": "Monday",
            "start_time": "10:00",
            "end_time": "11:00",
            "room": "R1",
            "faculty": "F2",
            "student_group": "G2",
        },
    ])


def test_calculate_conflicts(original_schedule):
    result = calculate_conflicts(original_schedule)

    assert result["room_conflicts"] == 1
    assert result["total_conflicts"] == 1


def test_calculate_total_time_shift(
    original_schedule,
    conflict_free_schedule,
):
    result = calculate_total_time_shift(
        original_schedule,
        conflict_free_schedule,
    )

    assert result == 30


def test_count_rescheduled_classes(
    original_schedule,
    conflict_free_schedule,
):
    result = count_rescheduled_classes(
        original_schedule,
        conflict_free_schedule,
    )

    assert result == 1


def test_evaluate_schedule(
    original_schedule,
    conflict_free_schedule,
):
    result = evaluate_schedule(
        original_schedule,
        conflict_free_schedule,
    )

    assert result["total_conflicts"] == 0
    assert result["rescheduled_classes"] == 1
    assert result["total_time_shift"] == 30


def test_rank_solutions(
    original_schedule,
    conflict_free_schedule,
):
    alternatives = {
        "Original": original_schedule,
        "Conflict-Free": conflict_free_schedule,
    }

    result = rank_solutions(
        original_schedule,
        alternatives,
    )

    assert result.iloc[0]["solution_name"] == "Conflict-Free"
    assert result.iloc[0]["rank"] == 1


def test_recommend_best_solution(
    original_schedule,
    conflict_free_schedule,
):
    alternatives = {
        "Original": original_schedule,
        "Conflict-Free": conflict_free_schedule,
    }

    result = recommend_best_solution(
        original_schedule,
        alternatives,
    )

    assert result["recommended_solution"] == "Conflict-Free"
    assert result["recommended_schedule"] is not None


def test_rank_solutions_without_alternatives(original_schedule):
    with pytest.raises(ValueError):
        rank_solutions(original_schedule, {})