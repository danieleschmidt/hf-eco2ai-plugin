#!/usr/bin/env python3
"""Check test coverage and enforce minimum thresholds."""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple


def parse_coverage_xml(coverage_file: Path) -> Tuple[float, Dict[str, float]]:
    """Parse coverage.xml and extract coverage metrics."""
    if not coverage_file.exists():
        raise FileNotFoundError(f"Coverage file not found: {coverage_file}")
    
    tree = ET.parse(coverage_file)
    root = tree.getroot()
    
    # Get overall coverage
    overall_coverage = float(root.attrib.get('line-rate', 0)) * 100
    
    # Get per-package coverage
    package_coverage = {}
    for package in root.findall('.//package'):
        package_name = package.attrib.get('name', 'unknown')
        line_rate = float(package.attrib.get('line-rate', 0)) * 100
        package_coverage[package_name] = line_rate
    
    return overall_coverage, package_coverage


def check_coverage_thresholds(
    overall_coverage: float,
    package_coverage: Dict[str, float],
    min_overall: float = 90.0,
    min_package: float = 85.0
) -> bool:
    """Check if coverage meets minimum thresholds."""
    success = True
    
    print(f"Overall coverage: {overall_coverage:.1f}%")
    if overall_coverage < min_overall:
        print(f"✗ Overall coverage {overall_coverage:.1f}% is below minimum {min_overall}%")
        success = False
    else:
        print(f"✓ Overall coverage {overall_coverage:.1f}% meets minimum {min_overall}%")
    
    print("\nPackage coverage:")
    for package, coverage in package_coverage.items():
        if package.startswith('src.'):
            package_short = package[4:]  # Remove 'src.' prefix
            print(f"  {package_short}: {coverage:.1f}%")
            
            if coverage < min_package:
                print(f"    ✗ Below minimum {min_package}%")
                success = False
            else:
                print(f"    ✓ Meets minimum {min_package}%")
    
    return success


def main() -> int:
    """Main coverage check function."""
    coverage_file = Path("coverage.xml")
    
    try:
        overall_coverage, package_coverage = parse_coverage_xml(coverage_file)
        
        print("Test Coverage Report")
        print("=" * 40)
        
        if check_coverage_thresholds(overall_coverage, package_coverage):
            print("\n✓ All coverage thresholds met")
            return 0
        else:
            print("\n✗ Coverage thresholds not met")
            return 1
            
    except FileNotFoundError:
        print("Coverage file not found. Run tests with coverage first:")
        print("  python -m pytest --cov=hf_eco2ai --cov-report=xml")
        return 1
    except Exception as e:
        print(f"Error checking coverage: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
