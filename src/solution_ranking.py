
import pandas as pd


# --------------------------------------------------
# 1. CALCULATE CONFLICTS
# --------------------------------------------------

from src.conflict_detector import (
    detect_room_conflicts,
    detect_faculty_conflicts,
    detect_student_group_conflicts,
)


def calculate_conflicts(schedule):
    """
    Calculate room, faculty, and student-group conflicts.
    """

    room_conflicts = detect_room_conflicts(schedule)
    faculty_conflicts = detect_faculty_conflicts(schedule)
    student_conflicts = detect_student_group_conflicts(schedule)

    return {
        "room_conflicts": len(room_conflicts),
        "faculty_conflicts": len(faculty_conflicts),
        "student_group_conflicts": len(student_conflicts),
        "total_conflicts": (
            len(room_conflicts)
            + len(faculty_conflicts)
            + len(student_conflicts)
        ),
    }


# --------------------------------------------------
# 2. VALIDATE SCHEDULE
# --------------------------------------------------

def validate_schedule(schedule):
    """
    Validate a timetable before ranking it.
    """

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
            f"Missing required columns: {missing_columns}"
        )

    if schedule.empty:
        raise ValueError("The timetable is empty.")

    if schedule["class_id"].duplicated().any():
        raise ValueError("Class IDs must be unique.")

    return True


# --------------------------------------------------
# 3. CALCULATE TOTAL TIME SHIFT
# --------------------------------------------------

def calculate_total_time_shift(original, alternative):
    """
    Calculate the total number of minutes by which
    class start times have changed.
    """

    validate_schedule(original)
    validate_schedule(alternative)

    original_times = original.set_index("class_id")
    alternative_times = alternative.set_index("class_id")

    if set(original_times.index) != set(alternative_times.index):
        raise ValueError(
            "Both schedules must contain the same class IDs."
        )

    total_shift = 0

    for class_id in original_times.index:

        original_time = pd.to_datetime(
            original_times.at[class_id, "start_time"],
            format="%H:%M",
        )

        alternative_time = pd.to_datetime(
            alternative_times.at[class_id, "start_time"],
            format="%H:%M",
        )

        difference = abs(
            (alternative_time - original_time).total_seconds()
        )

        total_shift += difference / 60

    return int(total_shift)


# --------------------------------------------------
# 4. COUNT RESCHEDULED CLASSES
# --------------------------------------------------

def count_rescheduled_classes(original, alternative):
    """
    Count how many classes have changed their
    start time or day compared with the original.
    """

    validate_schedule(original)
    validate_schedule(alternative)

    original_data = original.set_index("class_id")
    alternative_data = alternative.set_index("class_id")

    if set(original_data.index) != set(alternative_data.index):
        raise ValueError(
            "Both schedules must contain the same class IDs."
        )

    rescheduled = 0

    for class_id in original_data.index:

        original_row = original_data.loc[class_id]
        alternative_row = alternative_data.loc[class_id]

        if (
            str(original_row["start_time"])
            != str(alternative_row["start_time"])
            or str(original_row["day"])
            != str(alternative_row["day"])
        ):
            rescheduled += 1

    return rescheduled


# --------------------------------------------------
# 5. EVALUATE ONE SCHEDULE
# --------------------------------------------------

def evaluate_schedule(original, alternative):
    """
    Calculate the performance metrics of one
    alternative timetable.
    """

    validate_schedule(original)
    validate_schedule(alternative)

    conflicts = calculate_conflicts(alternative)

    total_shift = calculate_total_time_shift(
        original,
        alternative,
    )

    rescheduled_classes = count_rescheduled_classes(
        original,
        alternative,
    )

    return {
        "conflicts": conflicts,
        "total_conflicts": conflicts["total_conflicts"],
        "total_time_shift": total_shift,
        "rescheduled_classes": rescheduled_classes,
    }


# --------------------------------------------------
# 6. CALCULATE SOLUTION SCORE
# --------------------------------------------------

def calculate_solution_score(metrics):
    """
    Calculate a ranking score.

    Lower scores represent better schedules.

    We prioritize:
    1. Fewer conflicts.
    2. Fewer rescheduled classes.
    3. Less total time shifted.
    """

    conflict_weight = 10000
    reschedule_weight = 100
    time_shift_weight = 1

    score = (
        metrics["total_conflicts"] * conflict_weight
        + metrics["rescheduled_classes"] * reschedule_weight
        + metrics["total_time_shift"] * time_shift_weight
    )

    return score


# --------------------------------------------------
# 7. RANK MULTIPLE ALTERNATIVE SCHEDULES
# --------------------------------------------------

def rank_solutions(original_schedule, alternatives):
    """
    Evaluate and rank multiple alternative timetables.

    Parameters:
        original_schedule: Original timetable DataFrame.
        alternatives: Dictionary containing alternative
                      names and their timetable DataFrames.

    Returns:
        Ranked DataFrame with the best solution first.
    """

    validate_schedule(original_schedule)

    if not alternatives:
        raise ValueError(
            "At least one alternative schedule is required."
        )

    results = []

    for solution_name, alternative in alternatives.items():

        metrics = evaluate_schedule(
            original_schedule,
            alternative,
        )

        score = calculate_solution_score(metrics)

        results.append({
            "solution_name": solution_name,
            "room_conflicts": metrics["conflicts"]["room_conflicts"],
            "faculty_conflicts": metrics["conflicts"]["faculty_conflicts"],
            "student_group_conflicts": metrics["conflicts"]["student_group_conflicts"],
            "total_conflicts": metrics["total_conflicts"],
            "rescheduled_classes": metrics["rescheduled_classes"],
            "total_time_shift_minutes": metrics["total_time_shift"],
            "score": score,
        })

    ranking = pd.DataFrame(results)

    ranking = ranking.sort_values(
        by=[
            "total_conflicts",
            "rescheduled_classes",
            "total_time_shift_minutes",
        ],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    ranking.insert(
        0,
        "rank",
        range(1, len(ranking) + 1),
    )

    return ranking


# --------------------------------------------------
# 8. RECOMMEND THE BEST SOLUTION
# --------------------------------------------------

def recommend_best_solution(original_schedule, alternatives):
    """
    Return the highest-ranked alternative timetable.
    """

    ranking = rank_solutions(
        original_schedule,
        alternatives,
    )

    best_solution_name = ranking.iloc[0]["solution_name"]

    return {
        "recommended_solution": best_solution_name,
        "ranking": ranking,
        "recommended_schedule": alternatives[
            best_solution_name
        ].copy(),
    }
