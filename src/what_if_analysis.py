
import pandas as pd

from src.schedule_optimizer import optimize_schedule
from src.conflict_detector import (
    detect_room_conflicts,
    detect_faculty_conflicts,
    detect_student_group_conflicts,
)


# --------------------------------------------------
# 1. VALIDATE THE TIMETABLE
# --------------------------------------------------

def validate_scenario(schedule):

    required_columns = [
        "class_id",
        "course",
        "day",
        "start_time",
        "end_time",
        "room",
        "faculty",
        "student_group",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in schedule.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    if schedule.empty:
        raise ValueError("The timetable is empty.")

    if schedule["class_id"].duplicated().any():
        raise ValueError("Class IDs must be unique.")

    for _, row in schedule.iterrows():

        try:
            start = pd.to_datetime(
                row["start_time"],
                format="%H:%M"
            )

            end = pd.to_datetime(
                row["end_time"],
                format="%H:%M"
            )

        except (ValueError, TypeError):
            raise ValueError(
                "Time must use HH:MM format."
            )

        if end <= start:
            raise ValueError(
                f"Invalid duration for class {row['class_id']}."
            )

    return True


# --------------------------------------------------
# 2. COUNT ALL SCHEDULE CONFLICTS
# --------------------------------------------------

def count_schedule_conflicts(schedule):

    room_conflicts = detect_room_conflicts(schedule)

    faculty_conflicts = detect_faculty_conflicts(schedule)

    student_conflicts = detect_student_group_conflicts(
        schedule
    )

    room_count = len(room_conflicts)
    faculty_count = len(faculty_conflicts)
    student_count = len(student_conflicts)

    total_conflicts = (
        room_count + faculty_count + student_count
    )

    return {
        "room": room_count,
        "faculty": faculty_count,
        "student_group": student_count,
        "total": total_conflicts,
    }


# --------------------------------------------------
# 3. MODIFY A CLASS FOR WHAT-IF ANALYSIS
# --------------------------------------------------

def modify_class_scenario(
    schedule,
    class_id,
    new_start_time=None,
    new_day=None,
    new_room=None,
):

    """
    Create a modified copy of the timetable.

    Original timetable remains unchanged.
    Class duration is preserved.
    """

    validate_scenario(schedule)

    scenario = schedule.copy()

    matching_rows = scenario.index[
        scenario["class_id"] == class_id
    ]

    if len(matching_rows) == 0:

        raise ValueError(
            f"Class ID '{class_id}' was not found."
        )

    index = matching_rows[0]

    original_start = pd.to_datetime(
        scenario.at[index, "start_time"],
        format="%H:%M"
    )

    original_end = pd.to_datetime(
        scenario.at[index, "end_time"],
        format="%H:%M"
    )

    duration = original_end - original_start

    # Change class start time
    if new_start_time is not None:

        scenario.at[index, "start_time"] = new_start_time

    # Change class day
    if new_day is not None:

        scenario.at[index, "day"] = new_day

    # Change class room
    if new_room is not None:

        scenario.at[index, "room"] = new_room

    # Calculate the new end time
    try:

        modified_start = pd.to_datetime(
            scenario.at[index, "start_time"],
            format="%H:%M"
        )

    except (ValueError, TypeError):

        raise ValueError(
            "New start time must use HH:MM format."
        )

    modified_end = modified_start + duration

    if modified_end.date() != modified_start.date():

        raise ValueError(
            "Class cannot extend beyond the same day."
        )

    scenario.at[index, "end_time"] = (
        modified_end.strftime("%H:%M")
    )

    validate_scenario(scenario)

    return scenario


# --------------------------------------------------
# 4. RUN THE COMPLETE WHAT-IF SCENARIO
# --------------------------------------------------

def run_what_if_scenario(
    original_schedule,
    class_id,
    new_start_time=None,
    new_day=None,
    new_room=None,
    candidate_start_times=None,
):

    """
    Compare original, manually modified,
    and optimized timetables.
    """

    validate_scenario(original_schedule)

    # Original conflicts
    before_conflicts = count_schedule_conflicts(
        original_schedule
    )

    # Create modified timetable
    scenario_schedule = modify_class_scenario(
        schedule=original_schedule,
        class_id=class_id,
        new_start_time=new_start_time,
        new_day=new_day,
        new_room=new_room,
    )

    # Conflicts after manual modification
    scenario_conflicts = count_schedule_conflicts(
        scenario_schedule
    )

    # Optimize modified timetable
    optimization_result = optimize_schedule(
        scenario_schedule,
        candidate_start_times=candidate_start_times,
    )

    # IMPORTANT: Optimizer returns the result under "schedule"
    optimized_schedule = optimization_result.get(
        "schedule"
    )

    # Conflicts after optimization
    if optimized_schedule is not None:

        after_conflicts = count_schedule_conflicts(
            optimized_schedule
        )

    else:

        after_conflicts = None

    # Calculate conflict improvement
    if after_conflicts is not None:

        conflicts_resolved = (
            before_conflicts["total"]
            - after_conflicts["total"]
        )

    else:

        conflicts_resolved = None

    return {

        "original_schedule": original_schedule.copy(),

        "scenario_schedule": scenario_schedule,

        "optimization_result": optimization_result,

        "optimized_schedule": optimized_schedule,

        "before_conflicts": before_conflicts,

        "scenario_conflicts": scenario_conflicts,

        "after_conflicts": after_conflicts,

        "conflicts_resolved": conflicts_resolved,
    }
