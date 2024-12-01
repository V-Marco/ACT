import itertools
import json

def generate_combinations(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"Data from JSON: {data}")  # Debug statement
    
    variable_sets = data.get('variable_sets', [])
    for variable_set in variable_sets:
        keys = list(variable_set.keys())
        values = list(variable_set.values())

        # Create combinations for all key-value pairs
        for combination in itertools.product(*values):
            current_combination = dict(zip(keys, combination))
            yield current_combination

if __name__ == '__main__':
    json_file_path = 'constants_to_update.json'  # Specify your JSON file path here
    for combination in generate_combinations(json_file_path):
        print(json.dumps(combination))
        #print(f"Generated combination: {combination}")  # Debug statement
