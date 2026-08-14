import yaml
import json
import sys
import argparse
from pathlib import Path

def convert_yaml_to_json(yaml_path, json_path):
    # Read the YAML file
    with open(yaml_path, 'r') as yml_file:
        config_dict = yaml.safe_load(yml_file)
        
    # Save it as a JSON file
    if not json_path.endswith('.json'):
        json_path += '.json'
    with open(json_path, 'w') as json_file:
        json.dump(config_dict, json_file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_yaml", type=str, required=True)
    parser.add_argument("--out_json", type=str, required=True)
    args = parser.parse_args()

    convert_yaml_to_json(args.in_yaml, args.out_json)
