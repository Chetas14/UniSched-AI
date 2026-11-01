
import pandas as pd
from ortools.sat.python import cp_model


def _to_minutes(value):
    """Convert an HH:MM time into minutes after midnight."""

    try:
        hour, minute = map(int, str(value).split(":"))

        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError

        return hour * 60 + minute

    except (ValueError, AttributeError):
        raise ValueError(
            f"Invalid time '{value}'. Use HH:MM format."
        )


def _prepare_schedule(schedule, candidate_start_times):
    """Validate the timetable and prepare candidate start times."""

    required_columns = {
        "class_id",
        "course",
        "day",
        "start_time",
        "end_time",
        "room",
        "faculty",
        "student_group",
    }

    missing = required_columns - set(schedule.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if schedule.empty:
        return None, None

    data = schedule.copy().reset_index(drop=True)

    if data["class_id"].duplicated().any():
        raise ValueError("Each class_id must be unique.")

    if data[list(required_columns)].isnull().any().any():
        raise ValueError(
            "Required timetable fields cannot contain empty values."
        )

    data["_start"] = data["start_time"].map(_to_minutes)
    data["_end"] = data["end_time"].map(_to_minutes)

    if (data["_end"] <= data["_start"]).any():
        raise ValueError(
            "Each class must end after it starts and finish on the same day."
        )

    data["_duration"] = data["_end"] - data["_start"]

    if candidate_start_times is None:
        candidate_start_times = [
            f"{hour:02d}:{minute:02d}"
            for hour in range(8, 18)
            for minute in (0, 30)
        ]

    candidate_minutes = sorted(
        set(_to_minutes(t) for t in candidate_start_times)
    )

    if not candidate_minutes:
        raise ValueError(
            "Provide at least one candidate start time."
        )

    candidates = {}

    for i, row in data.iterrows():

        valid = [
            start
            for start in candidate_minutes
            if start + row["_duration"] <= 24 * 60
        ]

        if not valid:
            raise ValueError(
                f"No candidate time fits class {row['class_id']}."
            )

        candidates[i] = valid

    return data, candidates


def _build_model(data, candidates):
    """Create the CP-SAT scheduling model."""

    model = cp_model.CpModel()

    assigned = {}

    # One selected start time for every class.
    for i in range(len(data)):

        for start in candidates[i]:

            assigned[(i, start)] = model.new_bool_var(
                f"class_{i}_start_{start}"
            )

        model.add(
            sum(
                assigned[(i, start)]
                for start in candidates[i]
            ) == 1
        )

    # Minimize total time shifted from original start times.
    change_costs = []

    for i, row in data.iterrows():

        original_start = row["_start"]

        change_costs.extend(
            abs(start - original_start) * assigned[(i, start)]
            for start in candidates[i]
        )

    model.minimize(sum(change_costs))

    # Prevent overlapping classes from sharing resources.
    resource_columns = [
        "room",
        "faculty",
        "student_group",
    ]

    for i in range(len(data)):

        for j in range(i + 1, len(data)):

            first = data.iloc[i]
            second = data.iloc[j]

            if first["day"] != second["day"]:
                continue

            shared_resources = any(
                first[column] == second[column]
                for column in resource_columns
            )

            if not shared_resources:
                continue

            for start_i in candidates[i]:

                end_i = start_i + first["_duration"]

                for start_j in candidates[j]:

                    end_j = start_j + second["_duration"]

                    overlaps = (
                        start_i < end_j
                        and start_j < end_i
                    )

                    if overlaps:

                        model.add(
                            assigned[(i, start_i)]
                            + assigned[(j, start_j)]
                            <= 1
                        )

    return model, assigned


def _create_optimized_schedule(data, candidates, assigned, solver):
    """Convert the solver assignment into a timetable."""

    optimized = data.drop(
        columns=["_start", "_end", "_duration"]
    ).copy()

    for i in range(len(data)):

        chosen_start = next(
            start
            for start in candidates[i]
            if solver.value(assigned[(i, start)])
        )

        duration = data.iloc[i]["_duration"]

        chosen_end = chosen_start + duration

        optimized.at[i, "start_time"] = (
            f"{chosen_start // 60:02d}:"
            f"{chosen_start % 60:02d}"
        )

        optimized.at[i, "end_time"] = (
            f"{chosen_end // 60:02d}:"
            f"{chosen_end % 60:02d}"
        )

    return optimized


def _count_changed_classes(original, optimized):
    """Count classes whose start time changed."""

    return sum(
        str(original.iloc[i]["start_time"])
        != str(optimized.iloc[i]["start_time"])
        for i in range(len(original))
    )


def optimize_schedule(
    schedule: pd.DataFrame,
    candidate_start_times=None,
    max_time_seconds=10,
):
    """
    Find one conflict-free schedule using Google OR-Tools CP-SAT.

    Each class keeps its original day, room and duration.
    Its start time may change.

    Returns:
        status
        schedule
        message
    """

    if schedule.empty:

        return {
            "status": "NO_DATA",
            "schedule": None,
            "message": "The timetable contains no classes.",
        }

    data, candidates = _prepare_schedule(
        schedule,
        candidate_start_times,
    )

    model, assigned = _build_model(data, candidates)

    solver = cp_model.CpSolver()

    solver.parameters.max_time_in_seconds = max_time_seconds
    solver.parameters.num_search_workers = 8

    status_code = solver.solve(model)

    if status_code not in (
        cp_model.OPTIMAL,
        cp_model.FEASIBLE,
    ):

        return {
            "status": solver.status_name(status_code),
            "schedule": None,
            "message": (
                "No feasible schedule was found with the "
                "available candidate times and constraints."
            ),
        }

    optimized = _create_optimized_schedule(
        data,
        candidates,
        assigned,
        solver,
    )

    changed_classes = _count_changed_classes(
        schedule.reset_index(drop=True),
        optimized,
    )

    return {
        "status": solver.status_name(status_code),
        "schedule": optimized,
        "message": (
            f"Found a feasible schedule. "
            f"{changed_classes} class(es) changed their start time."
        ),
    }


def generate_alternative_schedules(
    schedule: pd.DataFrame,
    num_alternatives=3,
    candidate_start_times=None,
    max_time_seconds=10,
):
    """
    Generate multiple different feasible timetable alternatives.

    Each new solution excludes the exact assignment of all
    previously generated solutions.

    Classes retain their original day, room and duration.
    Only start times can change.

    Returns:
        status: Overall generation status
        alternatives: List of different timetable DataFrames
        message: Explanation of the result
    """

    if not isinstance(num_alternatives, int) or num_alternatives < 1:
        raise ValueError(
            "num_alternatives must be a positive integer."
        )

    if schedule.empty:

        return {
            "status": "NO_DATA",
            "alternatives": [],
            "message": "The timetable contains no classes.",
        }

    if max_time_seconds <= 0:
        raise ValueError(
            "max_time_seconds must be greater than zero."
        )

    data, candidates = _prepare_schedule(
        schedule,
        candidate_start_times,
    )

    model, assigned = _build_model(data, candidates)

    alternatives = []

    last_status = "UNKNOWN"

    for alternative_number in range(num_alternatives):

        solver = cp_model.CpSolver()

        solver.parameters.max_time_in_seconds = max_time_seconds
        solver.parameters.num_search_workers = 8

        status_code = solver.solve(model)

        last_status = solver.status_name(status_code)

        if status_code not in (
            cp_model.OPTIMAL,
            cp_model.FEASIBLE,
        ):
            break

        optimized = _create_optimized_schedule(
            data,
            candidates,
            assigned,
            solver,
        )

        alternatives.append(optimized)

        # Identify the exact assignment selected by this solution.
        selected_variables = []

        for i in range(len(data)):

            for start in candidates[i]:

                if solver.value(assigned[(i, start)]):

                    selected_variables.append(
                        assigned[(i, start)]
                    )

        # Exclude this exact solution so the next solve
        # must produce a different timetable.
        model.add(
            sum(selected_variables) <= len(data) - 1
        )

    if not alternatives:

        return {
            "status": last_status,
            "alternatives": [],
            "message": (
                "No feasible alternative schedules were found."
            ),
        }

    if len(alternatives) == num_alternatives:

        message = (
            f"Successfully generated {len(alternatives)} "
            f"different feasible schedules."
        )

    else:

        message = (
            f"Generated {len(alternatives)} different feasible "
            f"schedule(s) out of {num_alternatives} requested. "
            "No more feasible alternatives were found, "
            "or the solver reached its time limit."
        )

    return {
        "status": last_status,
        "alternatives": alternatives,
        "message": message,
    }
