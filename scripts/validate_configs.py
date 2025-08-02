#!/usr/bin/env python3
"""Validate configuration files."""

import json
import sys
from pathlib import Path
from typing import List

import yaml
from yaml.loader import SafeLoader


def validate_yaml_file(file_path: Path) -> bool:
    """Validate YAML file syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            yaml.load(f, Loader=SafeLoader)
        print(f"✓ {file_path} is valid YAML")
        return True
    except yaml.YAMLError as e:
        print(f"✗ {file_path} is invalid YAML: {e}")
        return False
    except Exception as e:
        print(f"✗ Error reading {file_path}: {e}")
        return False


def validate_json_file(file_path: Path) -> bool:
    """Validate JSON file syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json.load(f)
        print(f"✓ {file_path} is valid JSON")
        return True
    except json.JSONDecodeError as e:
        print(f"✗ {file_path} is invalid JSON: {e}")
        return False
    except Exception as e:
        print(f"✗ Error reading {file_path}: {e}")
        return False


def validate_toml_file(file_path: Path) -> bool:
    """Validate TOML file syntax."""
    try:
        import tomllib
        with open(file_path, 'rb') as f:
            tomllib.load(f)
        print(f"✓ {file_path} is valid TOML")
        return True
    except Exception as e:
        print(f"✗ {file_path} is invalid TOML: {e}")
        return False


def main(file_paths: List[str]) -> int:
    """Validate configuration files."""
    all_valid = True
    
    for file_path_str in file_paths:
        file_path = Path(file_path_str)
        
        if not file_path.exists():
            print(f"✗ {file_path} does not exist")
            all_valid = False
            continue
        
        suffix = file_path.suffix.lower()
        
        if suffix in ['.yml', '.yaml']:
            if not validate_yaml_file(file_path):
                all_valid = False
        elif suffix == '.json':
            if not validate_json_file(file_path):
                all_valid = False
        elif suffix == '.toml':
            if not validate_toml_file(file_path):
                all_valid = False
        else:
            print(f"? {file_path} - unknown file type, skipping")
    
    if all_valid:
        print("✓ All configuration files are valid")
        return 0
    else:
        print("✗ Some configuration files are invalid")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
