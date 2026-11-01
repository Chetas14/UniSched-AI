
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZIP_FILE = PROJECT_ROOT / "data" / "raw" / "pu-fal07-cs.zip"


def main():
    print("=" * 60)
    print("UniSched-AI: UniTime Dataset Inspection")
    print("=" * 60)

    if not ZIP_FILE.exists():
        raise FileNotFoundError(
            f"Dataset ZIP not found at:\n{ZIP_FILE}\n\n"
            "Check that pu-fal07-cs.zip is inside data/raw."
        )

    with zipfile.ZipFile(ZIP_FILE, "r") as archive:
        xml_files = [
            name for name in archive.namelist()
            if name.lower().endswith(".xml")
        ]

        if not xml_files:
            raise ValueError("No XML file found inside the ZIP.")

        xml_name = xml_files[0]
        print(f"\nXML file: {xml_name}")

        with archive.open(xml_name) as xml_file:
            root = ET.parse(xml_file).getroot()

    print(f"Dataset version: {root.get('version')}")
    print(f"Academic term: {root.get('term')}")
    print(f"Academic year: {root.get('year')}")

    print("\n--- Section counts ---")

    for section_name in (
        "rooms",
        "classes",
        "groupConstraints",
        "students",
    ):
        section = root.find(section_name)
        count = len(section) if section is not None else 0
        print(f"{section_name}: {count}")

    classes_section = root.find("classes")

    if classes_section is None:
        raise ValueError("The XML has no classes section.")

    print("\n--- Class data structure ---")

    for class_element in list(classes_section)[:3]:
        print(f"\nClass ID: {class_element.get('id')}")
        print(f"Class attributes: {class_element.attrib}")

        for child in class_element:
            print(f"  {child.tag}: {child.attrib}")

    print("\n--- Candidate time options ---")

    time_option_counts = Counter()
    room_option_counts = Counter()

    for class_element in classes_section:
        time_option_counts[len(class_element.findall("time"))] += 1
        room_option_counts[len(class_element.findall("room"))] += 1

    print("Number of time options per class:")
    for count, frequency in sorted(time_option_counts.items()):
        print(f"  {count} options: {frequency} classes")

    print("\nNumber of room options per class:")
    for count, frequency in sorted(room_option_counts.items()):
        print(f"  {count} options: {frequency} classes")

    print("\nInspection completed successfully.")
    print("Original dataset was not modified.")


if __name__ == "__main__":
    main()
