
from pathlib import Path
import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parent.parent
TIMETABLE_FILE = ROOT / "data" / "timetable_dataset.csv"
ROOMS_FILE = ROOT / "data" / "rooms.csv"
FACULTY_FILE = ROOT / "data" / "faculty_availability.csv"

def generate_resource_data():
    if not TIMETABLE_FILE.exists():
        raise FileNotFoundError("Timetable dataset was not found.")

    df = pd.read_csv(TIMETABLE_FILE)

    required_columns = ["room", "faculty"]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    days = "Monday;Tuesday;Wednesday;Thursday;Friday"

    # Generate room records
    rooms = sorted(df["room"].dropna().astype(str).unique())

    room_data = pd.DataFrame([
        {
            "room": room,
            "capacity": 60,
            "available_days": days,
            "availability_start": "08:00",
            "availability_end": "18:00",
            "data_is_sample": True
        }
        for room in rooms
    ])

    # Generate faculty records
    faculty_members = sorted(
        df["faculty"].dropna().astype(str).unique()
    )

    faculty_data = pd.DataFrame([
        {
            "faculty": faculty,
            "available_days": days,
            "availability_start": "08:00",
            "availability_end": "18:00",
            "data_is_sample": True
        }
        for faculty in faculty_members
    ])

    room_data.to_csv(ROOMS_FILE, index=False)
    faculty_data.to_csv(FACULTY_FILE, index=False)

    print("Resource sample files generated successfully!")
    print(f"Rooms created: {len(room_data)}")
    print(f"Faculty records created: {len(faculty_data)}")
    print(f"Room file: {ROOMS_FILE}")
    print(f"Faculty file: {FACULTY_FILE}")
    print("\nWARNING: All generated availability and capacity values are illustrative.")

if __name__ == "__main__":
    generate_resource_data()
