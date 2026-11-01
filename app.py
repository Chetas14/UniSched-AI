
import streamlit as st
import pandas as pd
from pathlib import Path

from src.conflict_detector import (
    detect_room_conflicts,
    detect_faculty_conflicts,
    detect_student_group_conflicts,
)

from src.schedule_optimizer import (
    optimize_schedule,
    generate_alternative_schedules,
)

from src.what_if_analysis import run_what_if_scenario

from src.solution_ranking import (
    rank_solutions,
    recommend_best_solution,
)

from src.resource_availability import (
    check_resource_availability,
    get_resource_summary,
)


# ==================================================
# CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="UniSched AI",
    page_icon="🎓",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

BENCHMARK_REPORT = BASE_DIR / "reports" / "unitime_conflicts.csv"
MODEL_EVALUATION_FILE = BASE_DIR / "data" / "model_evaluation.csv"

# Explainable AI report paths
XAI_DIR = BASE_DIR / "reports" / "explainable_ai"

PERMUTATION_IMPORTANCE_FILE = (
    XAI_DIR / "permutation_importance.csv"
)

SHAP_IMPORTANCE_FILE = (
    XAI_DIR / "shap_feature_importance.csv"
)

SHAP_SUMMARY_PLOT_FILE = (
    XAI_DIR / "shap_summary.png"
)


# ==================================================
# SAMPLE TIMETABLE
# ==================================================

def get_sample_schedule():

    return pd.DataFrame([
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
        {
            "class_id": "C3",
            "course": "Database Systems",
            "day": "Monday",
            "start_time": "09:45",
            "end_time": "10:15",
            "room": "R102",
            "faculty": "Dr. Sharma",
            "student_group": "CSE-C",
        },
        {
            "class_id": "C4",
            "course": "Computer Networks",
            "day": "Monday",
            "start_time": "10:00",
            "end_time": "11:00",
            "room": "R103",
            "faculty": "Dr. Patel",
            "student_group": "CSE-A",
        },
        {
            "class_id": "C5",
            "course": "Software Engineering",
            "day": "Tuesday",
            "start_time": "11:00",
            "end_time": "12:00",
            "room": "R101",
            "faculty": "Dr. Kumar",
            "student_group": "CSE-B",
        },
    ])


# ==================================================
# COMMON HELPERS
# ==================================================

def analyse_schedule(schedule):

    room = detect_room_conflicts(schedule)
    faculty = detect_faculty_conflicts(schedule)
    groups = detect_student_group_conflicts(schedule)

    return {
        "room": room,
        "faculty": faculty,
        "groups": groups,
        "total": len(room) + len(faculty) + len(groups),
    }


def display_conflict_table(conflicts, label):

    if conflicts.empty:
        st.success(f"No {label} conflicts found.")
        return

    st.error(f"{len(conflicts)} {label} conflict(s) found.")

    if "severity" in conflicts.columns:

        severity_counts = conflicts["severity"].value_counts()

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "High Severity",
            int(severity_counts.get("High", 0)),
        )

        col2.metric(
            "Medium Severity",
            int(severity_counts.get("Medium", 0)),
        )

        col3.metric(
            "Low Severity",
            int(severity_counts.get("Low", 0)),
        )

    st.dataframe(
        conflicts,
        use_container_width=True,
        hide_index=True,
    )


def get_candidate_times():

    return [
        f"{hour:02d}:{minute:02d}"
        for hour in range(8, 18)
        for minute in (0, 30)
    ]


# ==================================================
# BENCHMARK HELPERS
# ==================================================

def minutes_to_time(value):

    minutes = int(value)

    return (
        f"{(minutes // 60) % 24:02d}:"
        f"{minutes % 60:02d}"
    )


def load_benchmark_report():

    if not BENCHMARK_REPORT.exists():
        return None

    try:
        report = pd.read_csv(BENCHMARK_REPORT)

    except (
        OSError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ):
        return None

    required_columns = {
        "class_id_a",
        "class_id_b",
        "conflict_types",
        "start_minute_a",
        "end_minute_a",
        "start_minute_b",
        "end_minute_b",
        "shared_student_count",
        "shared_instructor_count",
    }

    if not required_columns.issubset(report.columns):
        return None

    return report


# ==================================================
# MODEL EVALUATION HELPERS
# ==================================================

def load_model_evaluation():

    if not MODEL_EVALUATION_FILE.exists():
        return None

    try:
        results = pd.read_csv(MODEL_EVALUATION_FILE)

    except (
        OSError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ):
        return None

    required_columns = {
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    }

    if not required_columns.issubset(results.columns):
        return None

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    ]

    for column in metric_columns:
        results[column] = pd.to_numeric(
            results[column],
            errors="coerce",
        )

    results = results.dropna(
        subset=["model"] + metric_columns
    )

    if results.empty:
        return None

    return results


# ==================================================
# EXPLAINABLE AI HELPERS
# ==================================================

def load_xai_results():

    if not (
        PERMUTATION_IMPORTANCE_FILE.exists()
        and SHAP_IMPORTANCE_FILE.exists()
    ):
        return None, None

    try:

        permutation = pd.read_csv(
            PERMUTATION_IMPORTANCE_FILE
        )

        shap_importance = pd.read_csv(
            SHAP_IMPORTANCE_FILE
        )

        return permutation, shap_importance

    except (
        OSError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ):
        return None, None


def find_feature_column(data):

    possible_names = [
        "feature",
        "features",
        "feature_name",
    ]

    for column in data.columns:

        if column.lower() in possible_names:
            return column

    return None


def find_importance_column(data):

    possible_columns = [
        column
        for column in data.select_dtypes(
            include="number"
        ).columns
    ]

    for column in possible_columns:

        name = column.lower()

        if (
            "importance" in name
            or "shap" in name
        ) and "std" not in name:

            return column

    return possible_columns[0] if possible_columns else None


# ==================================================
# HEADER
# ==================================================

st.title("🎓 UniSched AI")

st.subheader(
    "University Scheduling Conflict Detection & Resolution"
)

st.write(
    "Detect timetable conflicts, assess their severity, "
    "generate alternative schedules, compare solutions, "
    "and examine results from a real university benchmark."
)


# ==================================================
# SIDEBAR DATA SOURCE
# ==================================================

st.sidebar.header("Timetable Data")

data_source = st.sidebar.radio(
    "Choose data source",
    ["Sample Timetable", "Upload CSV"],
)


# ==================================================
# LOAD TIMETABLE
# ==================================================

if data_source == "Sample Timetable":

    schedule = get_sample_schedule()

else:

    uploaded_file = st.sidebar.file_uploader(
        "Upload timetable CSV",
        type=["csv"],
    )

    if uploaded_file is None:

        st.info("Upload a timetable CSV to continue.")

        st.stop()

    try:
        schedule = pd.read_csv(uploaded_file)

    except Exception as error:

        st.error(f"Could not read the CSV: {error}")

        st.stop()


# ==================================================
# VALIDATE TIMETABLE
# ==================================================

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

    st.error(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )

    st.stop()


if schedule.empty:

    st.warning("The timetable contains no classes.")

    st.stop()


if schedule[required_columns].isnull().any().any():

    st.error(
        "The timetable contains empty required values. "
        "Correct them before continuing."
    )

    st.stop()


if schedule["class_id"].duplicated().any():

    st.error("Class IDs must be unique.")

    st.stop()


try:
    analysis = analyse_schedule(schedule)

except (ValueError, TypeError) as error:

    st.error(f"Could not analyse the timetable: {error}")

    st.stop()


# ==================================================
# NAVIGATION
# ==================================================

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Conflict Detection",
        "Schedule Optimization",
        "What-If Analysis",
        "Solution Ranking",
        "Resource Availability",
        "Model Evaluation",
        "UniTime Benchmark",
        "Explainable AI (SHAP)",
    ],
)


# ==================================================
# 1. DASHBOARD
# ==================================================

if page == "Dashboard":

    st.header("📊 Dashboard Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Scheduled Classes", len(schedule))
    col2.metric("Room Conflicts", len(analysis["room"]))
    col3.metric("Faculty Conflicts", len(analysis["faculty"]))
    col4.metric("Student-Group Conflicts", len(analysis["groups"]))

    st.metric(
        "Total Conflict Instances",
        analysis["total"],
    )

    if analysis["total"] == 0:
        st.success("No conflicts detected.")
    else:
        st.warning(
            f"{analysis['total']} conflict instance(s) detected."
        )

    st.subheader("Conflict Severity Summary")

    all_conflicts = pd.concat(
        [
            analysis["room"],
            analysis["faculty"],
            analysis["groups"],
        ],
        ignore_index=True,
    )

    if (
        not all_conflicts.empty
        and "severity" in all_conflicts.columns
    ):

        severity_counts = (
            all_conflicts["severity"]
            .value_counts()
            .reindex(
                ["High", "Medium", "Low"],
                fill_value=0,
            )
        )

        severity_summary = severity_counts.rename_axis(
            "Severity"
        ).reset_index(
            name="Number of Conflicts"
        )

        st.bar_chart(
            severity_summary.set_index("Severity")
        )

    else:
        st.info("There are no conflicts to summarize.")

    st.subheader("Current Timetable")

    st.dataframe(
        schedule,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "The built-in timetable is demonstration data, "
        "not a real university dataset."
    )


# ==================================================
# 2. CONFLICT DETECTION
# ==================================================

elif page == "Conflict Detection":

    st.header("🔍 Conflict Detection")

    st.write(
        "Classes conflict when they overlap on the same day "
        "and share a room, faculty member, or student group."
    )

    st.info(
        "Severity is rule-based: Low = up to 15 minutes, "
        "Medium = 16–30 minutes, High = more than 30 minutes."
    )

    tabs = st.tabs(
        [
            "Room Conflicts",
            "Faculty Conflicts",
            "Student-Group Conflicts",
        ]
    )

    conflict_tables = [
        analysis["room"],
        analysis["faculty"],
        analysis["groups"],
    ]

    labels = [
        "room",
        "faculty",
        "student-group",
    ]

    for tab, conflicts, label in zip(
        tabs,
        conflict_tables,
        labels,
    ):

        with tab:
            display_conflict_table(conflicts, label)


# ==================================================
# 3. SCHEDULE OPTIMIZATION
# ==================================================

elif page == "Schedule Optimization":

    st.header("⚙️ Schedule Optimization")

    st.write(
        "The optimizer attempts to remove room, faculty, and "
        "student-group conflicts while preserving class durations "
        "and keeping each class on its original day."
    )

    st.info(
        "Candidate start times range from 08:00 to 17:30 "
        "in 30-minute intervals."
    )

    if st.button("🚀 Optimize Schedule", type="primary"):

        with st.spinner("Searching for a feasible schedule..."):

            try:

                result = optimize_schedule(
                    schedule,
                    candidate_start_times=get_candidate_times(),
                    max_time_seconds=10,
                )

            except (ValueError, TypeError) as error:

                st.error(f"Optimization failed: {error}")

                st.stop()

        optimized = result.get("schedule")

        if optimized is None:

            st.error(
                result.get(
                    "message",
                    "No feasible schedule was found.",
                )
            )

        else:

            optimized_analysis = analyse_schedule(optimized)

            st.success(
                result.get("message", "Optimization completed.")
            )

            st.write(
                "Solver status:",
                result.get("status", "Unknown"),
            )

            col1, col2, col3 = st.columns(3)

            col1.metric("Original Conflicts", analysis["total"])

            col2.metric(
                "Optimized Conflicts",
                optimized_analysis["total"],
            )

            rescheduled = sum(
                str(schedule.iloc[i]["start_time"])
                != str(optimized.iloc[i]["start_time"])
                for i in range(len(schedule))
            )

            col3.metric("Classes Rescheduled", rescheduled)

            if optimized_analysis["total"] == 0:

                st.success(
                    "All checked room, faculty, and student-group "
                    "conflicts have been resolved."
                )

            else:

                st.warning(
                    "Some conflicts remain. Review the results "
                    "before accepting this schedule."
                )

            st.subheader("Original Timetable")

            st.dataframe(
                schedule,
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Optimized Timetable")

            st.dataframe(
                optimized,
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Remaining Conflict Details")

            remaining_conflicts = pd.concat(
                [
                    optimized_analysis["room"],
                    optimized_analysis["faculty"],
                    optimized_analysis["groups"],
                ],
                ignore_index=True,
            )

            if remaining_conflicts.empty:
                st.success("No conflicts remain.")
            else:
                st.dataframe(
                    remaining_conflicts,
                    use_container_width=True,
                    hide_index=True,
                )

            st.download_button(
                "⬇️ Download Optimized Timetable",
                data=optimized.to_csv(index=False).encode("utf-8"),
                file_name="optimized_timetable.csv",
                mime="text/csv",
            )


# ==================================================
# 4. WHAT-IF ANALYSIS
# ==================================================

elif page == "What-If Analysis":

    st.header("🔄 What-If Analysis")

    st.write(
        "Experiment with changes to a class and examine how "
        "those changes affect timetable conflicts. You can "
        "then run the optimizer on the modified timetable."
    )

    st.info(
        "Your original timetable is not modified. Each scenario "
        "starts from the currently selected timetable."
    )

    st.subheader("Step 1: Select a Class")

    class_options = schedule["class_id"].tolist()

    selected_class = st.selectbox(
        "Choose the class to modify",
        options=class_options,
        format_func=lambda class_id: (
            f"{class_id} — "
            f"{schedule.loc[schedule['class_id'] == class_id, 'course'].iloc[0]}"
        ),
    )

    selected_row = schedule.loc[
        schedule["class_id"] == selected_class
    ].iloc[0]

    st.write("**Selected Class Details**")

    details_col1, details_col2, details_col3 = st.columns(3)

    details_col1.write(f"Course: {selected_row['course']}")
    details_col2.write(f"Day: {selected_row['day']}")
    details_col3.write(f"Room: {selected_row['room']}")

    st.write(
        f"Current time: {selected_row['start_time']} – "
        f"{selected_row['end_time']}"
    )

    st.divider()

    st.subheader("Step 2: Modify the Class")

    change_type = st.radio(
        "What would you like to change?",
        [
            "Start Time",
            "Day",
            "Room",
            "Multiple Properties",
        ],
        horizontal=True,
    )

    new_start_time = None
    new_day = None
    new_room = None

    available_days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    if change_type in ["Start Time", "Multiple Properties"]:

        time_options = get_candidate_times()

        current_time = str(selected_row["start_time"])

        time_index = (
            time_options.index(current_time)
            if current_time in time_options
            else 0
        )

        new_start_time = st.selectbox(
            "New Start Time",
            options=time_options,
            index=time_index,
        )

    if change_type in ["Day", "Multiple Properties"]:

        current_day = str(selected_row["day"])

        day_options = list(
            dict.fromkeys([current_day] + available_days)
        )

        new_day = st.selectbox(
            "New Day",
            options=day_options,
            index=0,
        )

    if change_type in ["Room", "Multiple Properties"]:

        available_rooms = sorted(
            schedule["room"].dropna().astype(str).unique().tolist()
        )

        current_room = str(selected_row["room"])

        room_index = (
            available_rooms.index(current_room)
            if current_room in available_rooms
            else 0
        )

        new_room = st.selectbox(
            "New Room",
            options=available_rooms,
            index=room_index,
        )

    st.divider()

    st.subheader("Step 3: Run What-If Analysis")

    if st.button("🔍 Analyze Scenario", type="primary"):

        try:

            result = run_what_if_scenario(
                original_schedule=schedule,
                class_id=selected_class,
                new_start_time=new_start_time,
                new_day=new_day,
                new_room=new_room,
                candidate_start_times=get_candidate_times(),
            )

            st.session_state["what_if_result"] = result
            st.session_state["what_if_class"] = selected_class

        except (ValueError, TypeError) as error:

            st.error(f"Could not run scenario: {error}")

        except Exception as error:

            st.error(f"An unexpected error occurred: {error}")

    if "what_if_result" in st.session_state:

        result = st.session_state["what_if_result"]

        st.divider()

        st.header("📊 Scenario Results")

        before = result["before_conflicts"]
        scenario = result["scenario_conflicts"]
        after = result["after_conflicts"]

        col1, col2, col3 = st.columns(3)

        col1.metric("Original Conflicts", before["total"])

        col2.metric(
            "After Manual Change",
            scenario["total"],
            delta=scenario["total"] - before["total"],
            delta_color="inverse",
        )

        if after is not None:

            col3.metric(
                "After Optimization",
                after["total"],
                delta=after["total"] - before["total"],
                delta_color="inverse",
            )

        else:
            col3.metric("After Optimization", "Not available")

        st.subheader("Conflict Breakdown")

        breakdown = pd.DataFrame({
            "Conflict Type": [
                "Room",
                "Faculty",
                "Student Group",
            ],
            "Original": [
                before["room"],
                before["faculty"],
                before["student_group"],
            ],
            "After Manual Change": [
                scenario["room"],
                scenario["faculty"],
                scenario["student_group"],
            ],
            "After Optimization": [
                after["room"] if after else 0,
                after["faculty"] if after else 0,
                after["student_group"] if after else 0,
            ],
        })

        st.dataframe(
            breakdown,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Modified Timetable")

        st.dataframe(
            result["scenario_schedule"],
            use_container_width=True,
            hide_index=True,
        )

        optimization_result = result["optimization_result"]
        optimized = result["optimized_schedule"]

        st.subheader("Optimization Result")

        st.write(
            "Solver status:",
            optimization_result.get("status", "Unknown"),
        )

        st.write(
            optimization_result.get(
                "message",
                "Optimization finished.",
            )
        )

        if optimized is not None:

            if after["total"] == 0:
                st.success(
                    "The optimized scenario has no detected conflicts."
                )
            else:
                st.warning(
                    f"{after['total']} conflict(s) remain after optimization."
                )

            st.subheader("Optimized Timetable")

            st.dataframe(
                optimized,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "⬇️ Download Scenario Timetable",
                data=optimized.to_csv(index=False).encode("utf-8"),
                file_name="what_if_optimized_timetable.csv",
                mime="text/csv",
                key="download_what_if",
            )

        else:

            st.warning(
                "The optimizer did not return a feasible schedule. "
                "The manually modified timetable is still available above."
            )

            st.download_button(
                "⬇️ Download Modified Timetable",
                data=result["scenario_schedule"].to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="what_if_modified_timetable.csv",
                mime="text/csv",
                key="download_what_if_modified",
            )

        st.caption(
            "The optimizer adjusts start times while preserving "
            "the manually selected day, room and class duration. "
            "It does not currently enforce room-capacity or faculty-"
            "availability constraints."
        )


# ==================================================
# 5. SOLUTION RANKING AND COMPARISON
# ==================================================

elif page == "Solution Ranking":

    st.header("🏆 Solution Ranking and Comparison")

    st.write(
        "Compare the original timetable with multiple optimized "
        "schedules and What-If alternatives. The system ranks "
        "schedules by conflict count, number of rescheduled classes, "
        "and total time shifted."
    )

    st.info(
        "Ranking priority: fewer conflicts first, then fewer "
        "rescheduled classes, then less total time shifted."
    )

    st.caption(
        "The optimizer generates distinct feasible schedules by "
        "excluding previously generated assignments."
    )

    if st.button(
        "🚀 Generate and Compare Solutions",
        type="primary",
    ):

        alternatives = {
            "Original Timetable": schedule.copy()
        }

        with st.spinner(
            "Generating three alternative schedules..."
        ):

            try:

                generation_result = generate_alternative_schedules(
                    schedule,
                    num_alternatives=3,
                    candidate_start_times=get_candidate_times(),
                    max_time_seconds=10,
                )

                generated_schedules = generation_result.get(
                    "alternatives",
                    [],
                )

                st.session_state["alternative_generation_status"] = (
                    generation_result.get("status", "Unknown")
                )

                st.session_state["alternative_generation_message"] = (
                    generation_result.get(
                        "message",
                        "Alternative generation completed.",
                    )
                )

                for index, alternative in enumerate(
                    generated_schedules,
                    start=1,
                ):

                    if alternative is not None:

                        alternatives[
                            f"Optimized Alternative {index}"
                        ] = alternative.copy()

                if len(generated_schedules) == 0:

                    st.warning(
                        "No optimized alternatives were generated. "
                        "The original timetable will still be evaluated."
                    )

                saved_what_if = st.session_state.get(
                    "what_if_result"
                )

                if saved_what_if is not None:

                    scenario_schedule = saved_what_if.get(
                        "scenario_schedule"
                    )

                    if scenario_schedule is not None:

                        alternatives["What-If Timetable"] = (
                            scenario_schedule.copy()
                        )

                ranking = rank_solutions(
                    schedule,
                    alternatives,
                )

                recommendation = recommend_best_solution(
                    schedule,
                    alternatives,
                )

                st.session_state["solution_ranking"] = ranking

                st.session_state["solution_alternatives"] = (
                    alternatives
                )

                st.session_state["solution_recommendation"] = (
                    recommendation
                )

                st.success(
                    "Solution comparison completed successfully!"
                )

            except Exception as error:

                st.error(
                    f"Could not compare solutions: {error}"
                )

    if "solution_ranking" in st.session_state:

        ranking = st.session_state["solution_ranking"]

        alternatives = st.session_state["solution_alternatives"]

        recommendation = st.session_state["solution_recommendation"]

        st.divider()

        st.subheader("⚙️ Alternative Generation Status")

        st.write(
            st.session_state.get(
                "alternative_generation_message",
                "No generation message available.",
            )
        )

        st.write(
            "Solver status:",
            st.session_state.get(
                "alternative_generation_status",
                "Unknown",
            ),
        )

        st.metric(
            "Schedules Compared",
            len(alternatives),
        )

        st.divider()

        st.subheader("🥇 Recommended Schedule")

        st.success(
            "Recommended solution: "
            + recommendation["recommended_solution"]
        )

        best_schedule = recommendation["recommended_schedule"]

        best_metrics = ranking.iloc[0]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Conflicts",
            int(best_metrics["total_conflicts"]),
        )

        col2.metric(
            "Classes Rescheduled",
            int(best_metrics["rescheduled_classes"]),
        )

        col3.metric(
            "Total Time Shift",
            f"{int(best_metrics['total_time_shift_minutes'])} min",
        )

        st.subheader("📊 Alternative Comparison")

        st.dataframe(
            ranking,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("📈 Ranking Metrics")

        chart_data = ranking.set_index("solution_name")[
            [
                "room_conflicts",
                "faculty_conflicts",
                "student_group_conflicts",
                "rescheduled_classes",
            ]
        ]

        st.bar_chart(chart_data)

        st.subheader("📅 Recommended Timetable")

        st.dataframe(
            best_schedule,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Inspect Alternatives")

        selected_solution = st.selectbox(
            "Choose a timetable to inspect",
            options=list(alternatives.keys()),
        )

        st.dataframe(
            alternatives[selected_solution],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Download Solution Ranking",
            data=ranking.to_csv(index=False).encode("utf-8"),
            file_name="solution_ranking.csv",
            mime="text/csv",
        )

        st.download_button(
            "⬇️ Download Recommended Timetable",
            data=best_schedule.to_csv(index=False).encode("utf-8"),
            file_name="recommended_timetable.csv",
            mime="text/csv",
        )

        st.caption(
            "The optimizer changes class start times while preserving "
            "the original day, room, and duration. The ranking evaluates "
            "conflicts, rescheduled classes, and total time shifted. "
            "Room capacity and faculty availability are not currently "
            "enforced by the optimizer."
        )


# ==================================================
# 6. RESOURCE AVAILABILITY
# ==================================================

elif page == "Resource Availability":

    st.header("🏫 Resource Availability Analysis")

    st.write(
        "Check whether rooms and faculty members are available "
        "during the scheduled class days and timings."
    )

    st.warning(
        "The current room capacities and availability schedules "
        "are illustrative sample data, not verified university data. "
        "Room capacity checks are skipped because student counts "
        "are not currently available in the timetable."
    )

    if st.button(
        "🔍 Analyze Resource Availability",
        type="primary",
    ):

        with st.spinner("Checking room and faculty availability..."):

            try:

                resource_results = check_resource_availability(
                    schedule
                )

                resource_summary = get_resource_summary(
                    resource_results
                )

                st.session_state["resource_results"] = (
                    resource_results
                )

                st.session_state["resource_summary"] = (
                    resource_summary
                )

                st.success(
                    "Resource availability analysis completed!"
                )

            except (
                ValueError,
                TypeError,
                FileNotFoundError,
            ) as error:

                st.error(
                    f"Resource analysis failed: {error}"
                )

    if "resource_results" in st.session_state:

        resource_results = st.session_state["resource_results"]

        resource_summary = st.session_state["resource_summary"]

        st.divider()

        st.subheader("📊 Availability Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Checks",
            resource_summary["total_checks"],
        )

        col2.metric(
            "Available",
            resource_summary["available"],
        )

        col3.metric(
            "Unavailable",
            resource_summary["unavailable"],
        )

        st.subheader("📈 Availability Distribution")

        chart_data = pd.DataFrame({
            "Status": ["Available", "Unavailable"],
            "Number of Checks": [
                resource_summary["available"],
                resource_summary["unavailable"],
            ],
        })

        st.bar_chart(
            chart_data.set_index("Status")
        )

        st.subheader("🏢 Room Availability")

        room_results = resource_results[
            resource_results["resource_type"] == "Room"
        ]

        st.dataframe(
            room_results,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("👨‍🏫 Faculty Availability")

        faculty_results = resource_results[
            resource_results["resource_type"] == "Faculty"
        ]

        st.dataframe(
            faculty_results,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("⚠️ Unavailable Resources")

        unavailable = resource_results[
            resource_results["status"] == "Unavailable"
        ]

        if unavailable.empty:

            st.success(
                "No resource availability issues were identified."
            )

        else:

            st.warning(
                f"{len(unavailable)} resource availability issue(s) found."
            )

            st.dataframe(
                unavailable,
                use_container_width=True,
                hide_index=True,
            )

        st.download_button(
            "⬇️ Download Resource Availability Report",
            data=resource_results.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="resource_availability_report.csv",
            mime="text/csv",
        )

        st.caption(
            "Availability is evaluated against the resource records "
            "in data/rooms.csv and data/faculty_availability.csv. "
            "Overlapping bookings are evaluated separately by the "
            "Conflict Detection module."
        )


# ==================================================
# 7. MODEL EVALUATION
# ==================================================

elif page == "Model Evaluation":

    st.header("🧠 Machine Learning Evaluation")

    st.write(
        "Compare the performance of the machine-learning models "
        "trained to classify timetable conflicts."
    )

    results = load_model_evaluation()

    if results is None:

        st.warning(
            "Model evaluation results could not be loaded. "
            "Run the following command in your project terminal:"
        )

        st.code("python -m src.train_model")

    else:

        metric_columns = [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
        ]

        invalid_metrics = (
            (results[metric_columns] < 0)
            | (results[metric_columns] > 1)
        ).any().any()

        if invalid_metrics:

            st.error(
                "One or more evaluation metrics fall outside "
                "the expected range of 0 to 1."
            )

        else:

            st.success(
                f"Loaded results for {len(results)} trained model(s)."
            )

            display_results = results.copy()

            display_results[metric_columns] = (
                display_results[metric_columns] * 100
            )

            display_results = display_results.rename(
                columns={
                    "model": "Model",
                    "accuracy": "Accuracy (%)",
                    "precision": "Precision (%)",
                    "recall": "Recall (%)",
                    "f1_score": "F1-score (%)",
                }
            )

            st.dataframe(
                display_results.style.format({
                    "Accuracy (%)": "{:.2f}%",
                    "Precision (%)": "{:.2f}%",
                    "Recall (%)": "{:.2f}%",
                    "F1-score (%)": "{:.2f}%",
                }),
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("📈 Performance Comparison")

            chart_data = results.set_index("model")[
                metric_columns
            ].rename(
                columns={
                    "accuracy": "Accuracy",
                    "precision": "Precision",
                    "recall": "Recall",
                    "f1_score": "F1-score",
                }
            )

            st.bar_chart(chart_data)

            best_model = (
                results.sort_values(
                    by="f1_score",
                    ascending=False,
                )
                .iloc[0]
            )

            st.subheader("🏆 Best Model by F1-score")

            col1, col2, col3 = st.columns(3)

            col1.metric("Best Model", str(best_model["model"]))

            col2.metric(
                "Accuracy",
                f"{best_model['accuracy']:.2%}",
            )

            col3.metric(
                "F1-score",
                f"{best_model['f1_score']:.2%}",
            )

            st.subheader("ℹ️ Understanding the Metrics")

            with st.expander("What do these metrics mean?"):

                st.write(
                    "**Accuracy:** The proportion of all predictions "
                    "that were correct."
                )

                st.write(
                    "**Precision:** Among classes predicted as conflicts, "
                    "the proportion that were actual conflicts."
                )

                st.write(
                    "**Recall:** Among actual conflicts, the proportion "
                    "correctly identified by the model."
                )

                st.write(
                    "**F1-score:** The harmonic mean of precision and "
                    "recall, balancing both metrics."
                )

            st.warning(
                "Research limitation: These results use synthetic "
                "timetable data. Although the train-test split separates "
                "entire schedules, the conflict labels are generated "
                "using deterministic rules and the model receives "
                "overlap-related features that directly encode those "
                "rules. Therefore, the reported scores do not establish "
                "real-world predictive performance. Evaluation on "
                "independently collected real-world schedules is needed."
            )

            st.download_button(
                "⬇️ Download Model Evaluation Results",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="model_evaluation.csv",
                mime="text/csv",
            )


# ==================================================
# 8. UNITIME BENCHMARK
# ==================================================

elif page == "UniTime Benchmark":

    st.header("📚 UniTime Benchmark Analysis")

    st.write(
        "Explore conflict pairs identified in the UniTime "
        "university course-timetabling benchmark."
    )

    st.caption(
        "This page reads the existing report generated by "
        "src/analyze_benchmark_conflicts.py. It does not rerun "
        "the benchmark analysis."
    )

    report = load_benchmark_report()

    if report is None:

        st.error(
            "The benchmark report could not be loaded. "
            "Check that reports/unitime_conflicts.csv exists "
            "and contains the expected columns."
        )

    else:

        total_pairs = len(report)

        room_conflicts = report[
            report["conflict_types"].astype(str).str.contains(
                "ROOM",
                case=False,
                na=False,
            )
        ]

        student_conflicts = report[
            report["conflict_types"].astype(str).str.contains(
                "STUDENT",
                case=False,
                na=False,
            )
        ]

        instructor_conflicts = report[
            report["conflict_types"].astype(str).str.contains(
                "INSTRUCTOR",
                case=False,
                na=False,
            )
        ]

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Scheduled Classes", "455")
        col2.metric("Conflict Pairs", total_pairs)
        col3.metric("Room Conflict Pairs", len(room_conflicts))
        col4.metric("Student Conflict Pairs", len(student_conflicts))

        st.subheader("Conflict Type Distribution")

        distribution = pd.DataFrame({
            "Conflict Type": [
                "Room",
                "Student",
                "Instructor",
            ],
            "Number of Pairs": [
                len(room_conflicts),
                len(student_conflicts),
                len(instructor_conflicts),
            ],
        })

        st.bar_chart(
            distribution.set_index("Conflict Type")
        )

        st.subheader("Benchmark Findings")

        st.write(
            f"- **{total_pairs}** conflict pairs are recorded "
            "in the report."
        )

        st.write(
            f"- **{len(room_conflicts)}** pairs involve room conflicts."
        )

        st.write(
            f"- **{len(student_conflicts)}** pairs involve "
            "shared students."
        )

        st.info(
            "Instructor-conflict results are incomplete: instructor "
            "data was recorded for only 34 of the 630 classes in "
            "the benchmark metadata. Zero recorded instructor "
            "conflicts must not be interpreted as proof that none exist."
        )

        st.subheader("Detailed Conflict Pairs")

        display_report = report.copy()

        for column in [
            "start_minute_a",
            "end_minute_a",
            "start_minute_b",
            "end_minute_b",
        ]:

            if column in display_report.columns:

                display_report[column] = (
                    display_report[column].apply(
                        lambda value: (
                            minutes_to_time(value)
                            if pd.notna(value)
                            else ""
                        )
                    )
                )

        display_report = display_report.rename(
            columns={
                "class_id_a": "Class A",
                "class_id_b": "Class B",
                "room_ids_a": "Rooms for Class A",
                "room_ids_b": "Rooms for Class B",
                "shared_room_ids": "Shared Rooms",
                "start_minute_a": "Class A Start",
                "end_minute_a": "Class A End",
                "start_minute_b": "Class B Start",
                "end_minute_b": "Class B End",
                "conflict_types": "Conflict Type",
                "shared_student_count": "Shared Students",
                "shared_instructor_count": "Shared Instructors",
            }
        )

        st.dataframe(
            display_report,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Download Benchmark Conflict Report",
            data=report.to_csv(index=False).encode("utf-8"),
            file_name="unitime_conflicts.csv",
            mime="text/csv",
        )


# ==================================================
# 9. EXPLAINABLE AI (SHAP)
# ==================================================

elif page == "Explainable AI (SHAP)":

    st.header("🧠 Explainable AI — SHAP")

    st.write(
        "Understand how different timetable features influence "
        "the machine-learning model's conflict predictions using "
        "SHAP and permutation feature importance."
    )

    st.info(
        "SHAP explains feature contributions to model predictions. "
        "Permutation importance measures how model performance "
        "changes when a feature is shuffled."
    )

    permutation, shap_importance = load_xai_results()

    if permutation is None or shap_importance is None:

        st.warning(
            "Explainable AI reports could not be loaded."
        )

        st.write(
            "Run the following command in your project terminal:"
        )

        st.code("python -m src.explainable_ai")

    else:

        st.success(
            "Explainable AI results loaded successfully."
        )

        # ------------------------------------------
        # PERMUTATION FEATURE IMPORTANCE
        # ------------------------------------------

        st.subheader(
            "1. Permutation Feature Importance"
        )

        st.write(
            "This analysis measures how much the model's "
            "performance decreases when each feature is randomly "
            "shuffled. Larger values indicate greater importance "
            "to the model's predictions."
        )

        st.dataframe(
            permutation,
            use_container_width=True,
            hide_index=True,
        )

        permutation_feature_column = find_feature_column(
            permutation
        )

        permutation_value_column = find_importance_column(
            permutation
        )

        if (
            permutation_feature_column is not None
            and permutation_value_column is not None
        ):

            permutation_chart = (
                permutation[
                    [
                        permutation_feature_column,
                        permutation_value_column,
                    ]
                ]
                .set_index(permutation_feature_column)
                .sort_values(
                    by=permutation_value_column,
                    ascending=False,
                )
            )

            st.bar_chart(permutation_chart)

        else:

            st.warning(
                "Could not identify the feature and importance "
                "columns in the permutation report."
            )

        st.divider()

        # ------------------------------------------
        # SHAP FEATURE IMPORTANCE
        # ------------------------------------------

        st.subheader(
            "2. SHAP Feature Importance"
        )

        st.write(
            "SHAP estimates how much each feature contributes "
            "to model predictions. Mean absolute SHAP values "
            "summarize the magnitude of each feature's influence."
        )

        st.dataframe(
            shap_importance,
            use_container_width=True,
            hide_index=True,
        )

        shap_feature_column = find_feature_column(
            shap_importance
        )

        shap_value_column = find_importance_column(
            shap_importance
        )

        if (
            shap_feature_column is not None
            and shap_value_column is not None
        ):

            shap_chart = (
                shap_importance[
                    [
                        shap_feature_column,
                        shap_value_column,
                    ]
                ]
                .set_index(shap_feature_column)
                .sort_values(
                    by=shap_value_column,
                    ascending=False,
                )
            )

            st.bar_chart(shap_chart)

        else:

            st.warning(
                "Could not identify the feature and importance "
                "columns in the SHAP report."
            )

        st.divider()

        # ------------------------------------------
        # SHAP SUMMARY PLOT
        # ------------------------------------------

        st.subheader(
            "3. SHAP Summary Plot"
        )

        if SHAP_SUMMARY_PLOT_FILE.exists():

            st.image(
                str(SHAP_SUMMARY_PLOT_FILE),
                caption="SHAP Feature Importance Summary",
                use_container_width=True,
            )

        else:

            st.warning(
                "SHAP summary plot was not found. "
                "Run the explainable AI script again."
            )

        st.divider()

        # ------------------------------------------
        # INTERPRETATION
        # ------------------------------------------

        st.subheader(
            "4. Model Interpretation"
        )

        if (
            shap_feature_column is not None
            and shap_value_column is not None
            and not shap_importance.empty
        ):

            top_feature = (
                shap_importance.sort_values(
                    by=shap_value_column,
                    ascending=False,
                )
                .iloc[0]
            )

            st.success(
                "Most influential feature according to SHAP: "
                + str(top_feature[shap_feature_column])
            )

            st.metric(
                "Highest Mean Absolute SHAP Importance",
                f"{float(top_feature[shap_value_column]):.6f}",
            )

        st.warning(
            "Research limitation: The model was trained using "
            "synthetic timetable data. The conflict labels are "
            "generated using deterministic rules, and overlap-related "
            "features directly encode those rules. Therefore, feature "
            "importance describes this trained model and dataset; "
            "it does not establish causal relationships or prove "
            "real-world predictive performance."
        )

        # ------------------------------------------
        # DOWNLOAD REPORTS
        # ------------------------------------------

        st.subheader("5. Download Explainability Reports")

        col1, col2 = st.columns(2)

        with col1:

            st.download_button(
                "⬇️ Download Permutation Importance",
                data=permutation.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="permutation_importance.csv",
                mime="text/csv",
            )

        with col2:

            st.download_button(
                "⬇️ Download SHAP Importance",
                data=shap_importance.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="shap_feature_importance.csv",
                mime="text/csv",
            )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "UniSched AI | INT395 Project | "
    "Conflict Detection and Constraint-Based Optimization"
)
