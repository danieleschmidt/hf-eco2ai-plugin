#!/usr/bin/env python3
"""Check version consistency across project files."""

import re
import sys
from pathlib import Path
from typing import Optional


def get_version_from_pyproject() -> Optional[str]:
    """Extract version from pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        return None
    
    content = pyproject_path.read_text()
    match = re.search(r'version = "([^"]+)"', content)
    return match.group(1) if match else None


def get_version_from_init() -> Optional[str]:
    """Extract version from __init__.py."""
    init_path = Path("src/hf_eco2ai/__init__.py")
    if not init_path.exists():
        return None
    
    content = init_path.read_text()
    match = re.search(r'__version__ = ["\']([^"\'\ ]+)["\']', content)
    return match.group(1) if match else None


def main() -> int:
    """Check version consistency."""
    pyproject_version = get_version_from_pyproject()
    init_version = get_version_from_init()
    
    print(f"pyproject.toml version: {pyproject_version}")
    print(f"__init__.py version: {init_version}")
    
    if pyproject_version is None:
        print("ERROR: Could not find version in pyproject.toml")
        return 1
    
    if init_version is None:
        print("ERROR: Could not find version in __init__.py")
        return 1
    
    if pyproject_version != init_version:
        print(f"ERROR: Version mismatch! {pyproject_version} != {init_version}")
        return 1
    
    print("✓ Version consistency check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
