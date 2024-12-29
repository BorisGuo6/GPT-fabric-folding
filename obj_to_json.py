import os
import json

def determine_cloth_type(folder_name):
    # Define a mapping for known cloth types based on folder prefixes
    cloth_type_map = {
        "TNSC": "Short-sleeve",
        "TCSC": "Short-sleeve",
        "TNNC": "No-sleeve",
        "TCLC": "Long-sleeve",
        "DLT": "No-sleeve",
        "DLSS": "Short-sleeve",
        "DSNS": "No-sleeve",
        "DSSS": "Short-sleeve",
        "DLNS": "No-sleeve",
        "PL": "Pants",
        "TCLO": "Long-sleeve",
        "TNLC": "Long-sleeve",
        "DLLS": "Long-sleeve",
        "DSLS": "Long-sleeve",
        "PS": "Pants",
        "DLG": "No-sleeve",
        "TNLO": "Long-sleeve",
        "SS": "No-sleeve",
        "TCNC": "No-sleeve",
        "THLO": "Long-sleeve",
        "THLC": "Long-sleeve",
        "SL": "No-sleeve",
        "THNC": "No-sleeve",
        "TCNO": "No-sleeve"
    }

    # Return the corresponding cloth type or "Unknown" if not in the map
    return cloth_type_map.get(folder_name, "Unknown")

def generate_cloth_json(root_folder, cloth_root_path):
    cloth_data = []

    # Traverse the directory structure
    for dirpath, _, filenames in os.walk(cloth_root_path):
        for filename in filenames:
            if filename.endswith('.obj'):
                # Extract the relative path
                relative_path = os.path.relpath(dirpath, cloth_root_path)
                path_parts = relative_path.split(os.sep)
                # print(relative_path)
                if len(path_parts) >= 2:
                    cloth_type = determine_cloth_type(path_parts[0])
                    cloth_name = os.path.join(path_parts[0], path_parts[1])

                    cloth_data.append({
                        "cloth_type": cloth_type,
                        "cloth_name": cloth_name
                    })

    # Add the cloth root entry
    cloth_data.insert(0, {"cloth_root": cloth_root_path})

    # Write to JSON
    with open("cloth_data.json", "w") as json_file:
        json.dump(cloth_data, json_file, indent=4)

    print("JSON file has been created: cloth_data.json")

# Usage
root_folder = "cloth_eval_data_all"
cloth_root_path = "./data/Cloth-Simulation/Assets/cloth_eval_data_all"
generate_cloth_json(root_folder, cloth_root_path)
