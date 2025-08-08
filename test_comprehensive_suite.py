#!/usr/bin/env python3
"""Comprehensive test suite for HF Eco2AI Plugin - All Quality Gates."""

import sys
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def run_test_suite(test_file, description):
    """Run a test suite and return results."""
    print(f"\n🧪 Running {description}...")
    print("=" * 60)
    
    try:
        result = subprocess.run([
            sys.executable, test_file
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        success = result.returncode == 0
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return success, result.stdout, result.stderr
    
    except Exception as e:
        print(f"Error running {test_file}: {e}")
        return False, "", str(e)


def main():
    """Run comprehensive quality gates."""
    print("🚀 COMPREHENSIVE QUALITY GATES - HF ECO2AI PLUGIN")
    print("=" * 80)
    
    test_suites = [
        ("test_basic_integration.py", "Basic Integration Tests (Generation 1)"),
        ("test_robustness_features.py", "Robustness & Reliability Tests (Generation 2)"),
        ("test_scaling_features.py", "Scaling & Performance Tests (Generation 3)"),
    ]
    
    results = []
    total_tests = 0
    total_passed = 0
    
    for test_file, description in test_suites:
        success, stdout, stderr = run_test_suite(test_file, description)
        results.append((description, success, stdout, stderr))
        
        # Parse test results
        if "tests passed" in stdout:
            try:
                # Extract "X/Y tests passed" pattern
                import re
                match = re.search(r'(\d+)/(\d+) tests passed', stdout)
                if match:
                    passed = int(match.group(1))
                    total = int(match.group(2))
                    total_passed += passed
                    total_tests += total
                else:
                    total_tests += 1
                    if success:
                        total_passed += 1
            except:
                total_tests += 1
                if success:
                    total_passed += 1
        else:
            total_tests += 1
            if success:
                total_passed += 1
    
    # Print summary
    print("\n" + "=" * 80)
    print("🎯 QUALITY GATES SUMMARY")
    print("=" * 80)
    
    for description, success, stdout, stderr in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} - {description}")
    
    print("\n" + "=" * 80)
    print(f"📊 OVERALL RESULTS: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("🎉 ALL QUALITY GATES PASSED!")
        print("\n✅ Generation 1: Basic functionality working")
        print("✅ Generation 2: Robust error handling & monitoring")  
        print("✅ Generation 3: Scalable performance optimization")
        print("\n🚀 Ready for production deployment!")
        return 0
    else:
        print("⚠️  SOME QUALITY GATES FAILED")
        failed_count = len(results) - sum(1 for _, success, _, _ in results if success)
        print(f"   {failed_count} test suite(s) failed")
        print("\n🔧 Review failed tests before deployment")
        return 1


if __name__ == "__main__":
    sys.exit(main())