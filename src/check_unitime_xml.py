
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

zip_path = Path("data/raw/pu-fal07-cs.zip")

with ZipFile(zip_path) as archive:
    xml_name = next(
        name for name in archive.namelist()
        if name.lower().endswith(".xml")
    )
    root = ET.fromstring(archive.read(xml_name))

classes = root.findall(".//class")

target_ids = {"947", "948", "949", "118", "119"}

print("Original XML class attributes")
print("=" * 60)

for element in classes:
    class_id = element.get("id")

    if class_id in target_ids:
        print(f"\nClass ID: {class_id}")
        print("Attributes:", element.attrib)
        print("Child tags:", [child.tag for child in element])
