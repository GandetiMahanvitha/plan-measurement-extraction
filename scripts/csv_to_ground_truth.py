import csv
import json
import sys


def convert(csv_path, json_path, plan_id):
    """Convert a CSV room/size table into the JSON structure used for plan validation."""
    measurements = []
    overall = None

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            space = row["Space"].strip()
            length = float(row["Length"])
            width = float(row["Width"])

            if space.lower() == "overall building size":
                overall = {"value_mm": length, "secondary_value_mm": width}
                continue

            measurements.append({
                "type": "room_dimension_pair",
                "value_mm": length,
                "secondary_value_mm": width,
                "element_name": space.upper(),
            })

    output = {"plan_id": plan_id, "measurements": measurements}
    if overall:
        output["overall_building_size"] = overall  # kept separately, see note below

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {len(measurements)} measurements to {json_path}")

if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2], sys.argv[3])