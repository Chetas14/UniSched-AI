# UniSched-AI
### Agentic AI for University Scheduling Conflict Detection & Resolution

UniSched-AI is an intelligent university scheduling assistant designed to detect timetable conflicts, analyze resource availability, generate alternative schedules, and recommend suitable scheduling solutions.

The project combines constraint-based optimization, machine learning, explainable AI, and an interactive Streamlit dashboard to support university timetable management.

---

## 1. Problem Statement

University timetables involve multiple interconnected resources, including:

- Classrooms and laboratories
- Faculty members
- Courses
- Student groups
- Time slots and working days

Changes in one part of a timetable can create conflicts across several resources. Identifying these conflicts and finding suitable alternatives manually can be time-consuming.

**UniSched-AI aims to automate this process** by detecting scheduling conflicts, evaluating resource availability, and generating feasible alternative schedules.

## 2. Project Objectives

- Detect room, faculty, and student-group conflicts.
- Classify conflicts according to severity.
- Analyze room and faculty availability.
- Generate alternative feasible schedules.
- Rank alternative schedules and recommend a suitable solution.
- Perform what-if scheduling analysis.
- Train and evaluate machine learning models.
- Explain model predictions using SHAP.
- Provide an interactive administrative dashboard.

## 3. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Explainable AI | SHAP |
| Optimization | Google OR-Tools (CP-SAT) |
| Visualization | Matplotlib, Streamlit |
| Testing | Pytest |
| Benchmark | UniTime benchmark dataset |
| Development Tools | VS Code, Git, GitHub |

## 4. Core Features

### Conflict Detection
Detects overlapping timetable assignments involving:
- Room conflicts
- Faculty conflicts
- Student-group conflicts

### Conflict Severity Classification
Classifies conflicts into:
- Low
- Medium
- High

### Resource Availability Analysis
Checks whether rooms and faculty members are available during scheduled days and time intervals.

### Schedule Optimization
Uses constraint programming to generate feasible timetable assignments while respecting scheduling constraints.

### Alternative Schedule Generation
Generates multiple different feasible timetable alternatives rather than returning only one solution.

### Solution Ranking
Evaluates and ranks alternatives based on:
- Number of conflicts
- Number of rescheduled classes
- Total time shift

### What-If Analysis
Allows scheduling changes to be evaluated before applying them to the original timetable.

### Machine Learning
Trains and compares multiple classification models for timetable conflict prediction.

### Explainable AI
Uses SHAP and permutation feature importance to examine which features influence model predictions.

### Admin Dashboard
Provides an interactive Streamlit interface for accessing project modules and viewing results.

## 5. Machine Learning Pipeline

The project uses a separate machine learning dataset containing 10,000 timetable records generated from 20 schedules.

The dataset contains:
- 5,200 records without conflicts
- 4,800 records with conflicts

The training pipeline includes:
1. Data preparation
2. Feature engineering
3. Preprocessing
4. Train-test splitting by schedule
5. Model training
6. Model comparison
7. Evaluation
8. Model saving

The data is split by schedule using GroupShuffleSplit to avoid placing records from the same schedule in both training and testing sets.

### Models Evaluated

- Logistic Regression
- Random Forest
- Extra Trees

### Evaluation Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 100% | 100% | 100% | 100% |
| Random Forest | 100% | 100% | 100% | 100% |
| Extra Trees | 100% | 100% | 100% | 100% |

The current best model is Logistic Regression.

**Important:** These results are from synthetic data. The labels are generated using deterministic scheduling-conflict rules, and some model features directly encode those rules. Therefore, these scores do not establish equivalent predictive performance on real university timetables.

## 6. Explainable AI (SHAP)

The project integrates SHAP to investigate feature contributions to model predictions.

The explainability module generates:
- Permutation feature importance
- SHAP feature importance
- SHAP summary visualization

The analysis currently indicates that `overlapping_resource_classes` is the dominant feature.

This is consistent with the synthetic label-generation process and should not be interpreted as evidence of causal relationships in real-world university scheduling.

## 7. Project Structure

```text
UniSched-AI/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── timetable_dataset.csv
│   ├── ml_timetable_dataset.csv
│   ├── rooms.csv
│   ├── faculty_availability.csv
│   ├── model_evaluation.csv
│   └── raw/
│
├── src/
│   ├── conflict_detector.py
│   ├── schedule_optimizer.py
│   ├── resource_availability.py
│   ├── solution_ranking.py
│   ├── what_if_analysis.py
│   ├── explainable_ai.py
│   ├── train_model.py
│   └── generate_resource_data.py
│
├── tests/
│   ├── test_conflict_detector.py
│   ├── test_resource_availability.py
│   ├── test_schedule_optimizer.py
│   ├── test_solution_ranking.py
│   └── test_what_if_analysis.py
│
├── models/
│   └── best_conflict_model.joblib
│
└── reports/
    └── explainable_ai/
        ├── permutation_importance.csv
        ├── shap_feature_importance.csv
        └── shap_summary.png
```

*The structure above describes the main project components. Update filenames if your final repository differs.*

## 8. Installation and Setup

### Prerequisites

- Python 3.14.4 (the development environment used for this project)
- pip
- Git

### Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd UniSched-AI
```

### Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

The application should open in your browser at:

```text
http://localhost:8501
```

## 9. Running Tests

Run the complete test suite:

```bash
python -m pytest -v
```

Latest verified result:

**22 tests passed.**

## 10. Benchmark Dataset

The project includes a UniTime benchmark dataset for additional timetable evaluation.

The current benchmark conversion identified:
- 630 classes
- 455 selected time-and-room assignments
- 19 conflict pairs in the evaluated benchmark
- 6 room conflict pairs
- 13 student-group conflict pairs

Faculty information was available for only a subset of benchmark classes, so faculty-related benchmark evaluation is limited.

## 11. Dataset Limitations

The project uses synthetic timetable data for development and model training.

The generated room and faculty availability records are illustrative sample data, not verified institutional records.

For real-world deployment, the system would require:
- Actual university timetable data
- Verified room capacities
- Real faculty availability
- Student enrollment and group information
- Institutional scheduling rules

## 12. Future Enhancements

- Integration with real university timetable systems.
- More realistic institutional scheduling constraints.
- Improved validation using real-world datasets.
- Public cloud deployment.
- User authentication and role-based access.
- Export optimized timetables to CSV and Excel.
- Further optimization and evaluation on larger institutional schedules.

## 13. Project Status

The core scheduling, optimization, machine learning, explainability, and dashboard modules have been implemented.

The latest automated test run passed all 22 tests.

GitHub finalization, project documentation, and public deployment remain part of the remaining work.

---

## Author

**S Chetas**

B.Tech – Computer Science and Engineering (AI & ML)

Project: INT395 – Agentic AI for University Scheduling Conflict Detection & Resolution
