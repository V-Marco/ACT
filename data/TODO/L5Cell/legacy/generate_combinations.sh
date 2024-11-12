import itertools
import json

def generate_combinations(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    variable_sets = data.get('variable_sets', [])
    for variable_set in variable_sets:
        keys = variable_set.keys()
        values = variable_set.values()

        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))

if __name__ == "__main__":
    for combination in generate_combinations('constants_to_update.json'):
        print(json.dumps(combination))
