import json
import re
import sys

def update_constants(json_data):
    print(f"Received json_data: {json_data}")  # Debug statement
    
    with open("constants.py", "r") as f:
        lines = f.readlines()

    with open("constants.py", "w") as f:
        for line in lines:
            matched = False
            for key, value in json_data.items():
                if re.match(f"{key}\s*=", line):
                    f.write(f"{key} = {json.dumps(value)}\n")
                    print(f"Updated {key} to {value}")  # Debug statement
                    matched = True
                    break
            if not matched:
                f.write(line)
                
    print("Updated data:", json_data)


if __name__ == '__main__':
    json_str = sys.argv[1]
    print(f"Received json_str: {json_str}")  # Debug statement
    #print(f"The type of json_str is {type(json_str)}")
    #print(f"The content of json_str is {json_str}")
    try:
        json_data = json.loads(json_str)
        update_constants(json_data)
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")