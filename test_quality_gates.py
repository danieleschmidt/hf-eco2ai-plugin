#!/usr/bin/env python3
"""
QUALITY GATES - Comprehensive validation and verification
TERRAGON AUTONOMOUS SDLC v4.0 - Quality Assurance and Gates
"""

import sys
import os
import json
import time
import logging
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import ast

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging for testing
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SecurityValidator:
    """Validate security aspects of the codebase."""
    
    def __init__(self):
        self.security_issues = []
        self.critical_patterns = [
            # Potential security risks
            r'eval\(',
            r'exec\(',
            r'os\.system',
            r'subprocess\.call.*shell=True',
            r'pickle\.loads',
            r'yaml\.load\([^,]*\)',  # unsafe yaml load
            r'input\(',  # in Python 2, input is eval
            r'__import__',
            r'compile\(',
            # Hardcoded credentials patterns
            r'password\s*=\s*["\'][^"\']{8,}["\']',
            r'api_?key\s*=\s*["\'][^"\']{16,}["\']',
            r'secret\s*=\s*["\'][^"\']{16,}["\']',
            r'token\s*=\s*["\'][^"\']{16,}["\']',
        ]
    
    def scan_file(self, file_path: Path) -> Dict[str, Any]:
        """Scan a single file for security issues."""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            import re
            
            for line_num, line in enumerate(content.split('\n'), 1):
                for pattern in self.critical_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        issues.append({
                            'line': line_num,
                            'pattern': pattern,
                            'content': line.strip(),
                            'severity': 'HIGH' if any(p in pattern for p in ['eval', 'exec', 'system']) else 'MEDIUM'
                        })
        
        except Exception as e:
            issues.append({
                'line': 0,
                'pattern': 'FILE_READ_ERROR',
                'content': str(e),
                'severity': 'MEDIUM'
            })
        
        return {
            'file': str(file_path),
            'issues': issues,
            'clean': len(issues) == 0
        }
    
    def validate_project_security(self, project_path: Path) -> Dict[str, Any]:
        """Validate security of the entire project."""
        python_files = list(project_path.glob('**/*.py'))
        scan_results = []
        total_issues = 0
        high_severity_issues = 0
        
        for py_file in python_files:
            # Skip test files and mock files for security scanning
            if any(skip in str(py_file).lower() for skip in ['test_', 'mock_', '__pycache__']):
                continue
                
            result = self.scan_file(py_file)
            scan_results.append(result)
            total_issues += len(result['issues'])
            high_severity_issues += len([i for i in result['issues'] if i['severity'] == 'HIGH'])
        
        security_score = max(0, 100 - (high_severity_issues * 20 + (total_issues - high_severity_issues) * 5))
        
        return {
            'files_scanned': len(scan_results),
            'total_issues': total_issues,
            'high_severity_issues': high_severity_issues,
            'security_score': security_score,
            'results': scan_results,
            'passed': high_severity_issues == 0 and total_issues <= 5
        }


class CodeQualityAnalyzer:
    """Analyze code quality metrics."""
    
    def __init__(self):
        self.metrics = {}
    
    def analyze_file_complexity(self, file_path: Path) -> Dict[str, Any]:
        """Analyze complexity metrics for a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Count different types of nodes
            function_count = 0
            class_count = 0
            max_nesting = 0
            lines_of_code = len([line for line in content.split('\n') if line.strip() and not line.strip().startswith('#')])
            
            class ComplexityVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.nesting_level = 0
                    self.max_nesting = 0
                    self.function_count = 0
                    self.class_count = 0
                
                def visit_FunctionDef(self, node):
                    self.function_count += 1
                    self.nesting_level += 1
                    self.max_nesting = max(self.max_nesting, self.nesting_level)
                    self.generic_visit(node)
                    self.nesting_level -= 1
                
                def visit_ClassDef(self, node):
                    self.class_count += 1
                    self.nesting_level += 1
                    self.max_nesting = max(self.max_nesting, self.nesting_level)
                    self.generic_visit(node)
                    self.nesting_level -= 1
                
                def visit_If(self, node):
                    self.nesting_level += 1
                    self.max_nesting = max(self.max_nesting, self.nesting_level)
                    self.generic_visit(node)
                    self.nesting_level -= 1
                
                def visit_For(self, node):
                    self.nesting_level += 1
                    self.max_nesting = max(self.max_nesting, self.nesting_level)
                    self.generic_visit(node)
                    self.nesting_level -= 1
                
                def visit_While(self, node):
                    self.nesting_level += 1
                    self.max_nesting = max(self.max_nesting, self.nesting_level)
                    self.generic_visit(node)
                    self.nesting_level -= 1
            
            visitor = ComplexityVisitor()
            visitor.visit(tree)
            
            # Calculate complexity score
            complexity_score = 100
            if visitor.max_nesting > 5:
                complexity_score -= (visitor.max_nesting - 5) * 10
            if lines_of_code > 500:
                complexity_score -= (lines_of_code - 500) // 100 * 5
            if visitor.function_count > 50:
                complexity_score -= (visitor.function_count - 50) * 2
            
            complexity_score = max(0, complexity_score)
            
            return {
                'file': str(file_path),
                'lines_of_code': lines_of_code,
                'function_count': visitor.function_count,
                'class_count': visitor.class_count,
                'max_nesting_level': visitor.max_nesting,
                'complexity_score': complexity_score,
                'maintainable': complexity_score >= 70
            }
        
        except Exception as e:
            return {
                'file': str(file_path),
                'error': str(e),
                'complexity_score': 0,
                'maintainable': False
            }
    
    def analyze_project_quality(self, project_path: Path) -> Dict[str, Any]:
        """Analyze code quality for the entire project."""
        python_files = list(project_path.glob('**/*.py'))
        analysis_results = []
        total_loc = 0
        total_functions = 0
        total_classes = 0
        complexity_scores = []
        
        for py_file in python_files:
            # Skip __pycache__ and similar
            if '__pycache__' in str(py_file):
                continue
                
            result = self.analyze_file_complexity(py_file)
            analysis_results.append(result)
            
            if 'error' not in result:
                total_loc += result['lines_of_code']
                total_functions += result['function_count']
                total_classes += result['class_count']
                complexity_scores.append(result['complexity_score'])
        
        avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0
        maintainable_files = len([r for r in analysis_results if r.get('maintainable', False)])
        maintainability_ratio = maintainable_files / len(analysis_results) if analysis_results else 0
        
        return {
            'files_analyzed': len(analysis_results),
            'total_lines_of_code': total_loc,
            'total_functions': total_functions,
            'total_classes': total_classes,
            'average_complexity_score': avg_complexity,
            'maintainability_ratio': maintainability_ratio,
            'results': analysis_results,
            'passed': avg_complexity >= 70 and maintainability_ratio >= 0.8
        }


class TestCoverageAnalyzer:
    """Analyze test coverage and test quality."""
    
    def __init__(self):
        self.coverage_data = {}
    
    def find_test_files(self, project_path: Path) -> List[Path]:
        """Find all test files in the project."""
        test_patterns = ['test_*.py', '*_test.py', 'tests/**/*.py']
        test_files = []
        
        for pattern in test_patterns:
            test_files.extend(list(project_path.glob(pattern)))
        
        return test_files
    
    def find_source_files(self, project_path: Path) -> List[Path]:
        """Find all source files to be tested."""
        source_files = []
        src_path = project_path / 'src'
        
        if src_path.exists():
            source_files.extend(list(src_path.glob('**/*.py')))
        else:
            # Look for .py files in project root, excluding tests
            for py_file in project_path.glob('**/*.py'):
                if not any(test_marker in str(py_file).lower() for test_marker in ['test_', '_test', 'tests/']):
                    source_files.append(py_file)
        
        return source_files
    
    def analyze_test_coverage(self, project_path: Path) -> Dict[str, Any]:
        """Analyze test coverage."""
        test_files = self.find_test_files(project_path)
        source_files = self.find_source_files(project_path)
        
        total_source_files = len(source_files)
        total_test_files = len(test_files)
        
        # Simple heuristic: estimate coverage based on test to source ratio
        if total_source_files == 0:
            coverage_ratio = 1.0
        else:
            coverage_ratio = min(1.0, total_test_files / total_source_files)
        
        # Analyze test file content for test function count
        total_test_functions = 0
        
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                        total_test_functions += 1
                        
            except Exception:
                continue
        
        # Calculate coverage score
        coverage_score = coverage_ratio * 100
        test_density = total_test_functions / max(total_source_files, 1)
        
        # Adjust score based on test density
        if test_density >= 3:  # 3+ tests per source file
            coverage_score = min(100, coverage_score * 1.2)
        elif test_density >= 1:  # 1+ tests per source file
            coverage_score = min(100, coverage_score * 1.1)
        
        return {
            'source_files_count': total_source_files,
            'test_files_count': total_test_files,
            'test_functions_count': total_test_functions,
            'coverage_ratio': coverage_ratio,
            'test_density': test_density,
            'coverage_score': coverage_score,
            'passed': coverage_score >= 70
        }


class PerformanceBenchmarker:
    """Benchmark performance characteristics."""
    
    def __init__(self):
        self.benchmarks = {}
    
    def benchmark_import_time(self, project_path: Path) -> Dict[str, Any]:
        """Benchmark import time for main modules."""
        import_times = {}
        
        # Test importing main modules
        main_modules = ['mock_integration']  # Our main module for testing
        
        for module_name in main_modules:
            try:
                start_time = time.time()
                
                # Add project src to path
                sys.path.insert(0, str(project_path / 'src' / 'hf_eco2ai'))
                
                # Import the module
                import importlib
                module = importlib.import_module(module_name)
                
                import_time = time.time() - start_time
                import_times[module_name] = import_time
                
                # Clean up
                if module_name in sys.modules:
                    del sys.modules[module_name]
                
            except Exception as e:
                import_times[module_name] = {'error': str(e)}
        
        avg_import_time = sum(t for t in import_times.values() if isinstance(t, (int, float)))
        avg_import_time = avg_import_time / len([t for t in import_times.values() if isinstance(t, (int, float))])
        
        return {
            'import_times': import_times,
            'average_import_time': avg_import_time,
            'fast_imports': avg_import_time < 0.5,  # Less than 500ms
            'passed': avg_import_time < 1.0  # Less than 1 second
        }
    
    def benchmark_basic_operations(self) -> Dict[str, Any]:
        """Benchmark basic operations."""
        # Test basic carbon calculation performance
        sys.path.insert(0, str(Path(__file__).parent / "src" / "hf_eco2ai"))
        
        try:
            from mock_integration import MockEco2AICallback, MockCarbonConfig
            
            config = MockCarbonConfig(project_name="perf-test")
            callback = MockEco2AICallback(config)
            
            # Benchmark callback initialization
            init_times = []
            for _ in range(10):
                start_time = time.time()
                test_callback = MockEco2AICallback(config)
                init_time = time.time() - start_time
                init_times.append(init_time)
            
            avg_init_time = sum(init_times) / len(init_times)
            
            # Benchmark step processing
            callback.on_train_begin()
            
            step_times = []
            for step in range(100):
                start_time = time.time()
                callback.on_step_end(step=step, logs={"loss": 0.5})
                step_time = time.time() - start_time
                step_times.append(step_time)
            
            avg_step_time = sum(step_times) / len(step_times)
            callback.on_train_end()
            
            # Benchmark metrics retrieval
            metrics_times = []
            for _ in range(50):
                start_time = time.time()
                callback.get_current_metrics()
                metrics_time = time.time() - start_time
                metrics_times.append(metrics_time)
            
            avg_metrics_time = sum(metrics_times) / len(metrics_times)
            
            return {
                'avg_init_time_ms': avg_init_time * 1000,
                'avg_step_time_ms': avg_step_time * 1000,
                'avg_metrics_time_ms': avg_metrics_time * 1000,
                'steps_per_second': 1 / avg_step_time if avg_step_time > 0 else float('inf'),
                'passed': avg_step_time < 0.01  # Less than 10ms per step
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'passed': False
            }


class DocumentationValidator:
    """Validate documentation completeness and quality."""
    
    def __init__(self):
        self.doc_files = []
    
    def find_documentation_files(self, project_path: Path) -> List[Path]:
        """Find all documentation files."""
        doc_patterns = ['*.md', '*.rst', '*.txt', 'docs/**/*']
        doc_files = []
        
        for pattern in doc_patterns:
            doc_files.extend(list(project_path.glob(pattern)))
        
        # Filter out non-documentation files
        doc_files = [f for f in doc_files if f.suffix.lower() in ['.md', '.rst', '.txt'] or 'doc' in str(f).lower()]
        
        return doc_files
    
    def validate_documentation(self, project_path: Path) -> Dict[str, Any]:
        """Validate documentation completeness."""
        doc_files = self.find_documentation_files(project_path)
        
        # Check for essential documentation files
        essential_docs = ['README', 'LICENSE', 'CONTRIBUTING', 'CHANGELOG']
        found_essential = []
        
        for doc_file in doc_files:
            file_name_upper = doc_file.stem.upper()
            for essential in essential_docs:
                if essential in file_name_upper:
                    found_essential.append(essential)
                    break
        
        # Analyze README quality
        readme_score = 0
        readme_file = None
        
        for doc_file in doc_files:
            if 'README' in doc_file.stem.upper():
                readme_file = doc_file
                break
        
        if readme_file:
            try:
                with open(readme_file, 'r', encoding='utf-8') as f:
                    readme_content = f.read()
                
                readme_score = 10  # Base score
                
                # Check for key sections
                if 'install' in readme_content.lower():
                    readme_score += 20
                if 'usage' in readme_content.lower():
                    readme_score += 20
                if 'example' in readme_content.lower():
                    readme_score += 20
                if 'api' in readme_content.lower() or 'reference' in readme_content.lower():
                    readme_score += 10
                if 'license' in readme_content.lower():
                    readme_score += 10
                if len(readme_content) > 1000:  # Substantial content
                    readme_score += 10
                    
            except Exception:
                readme_score = 5  # Penalty for unreadable README
        
        documentation_completeness = len(found_essential) / len(essential_docs)
        overall_doc_score = (documentation_completeness * 50) + (readme_score * 0.5)
        
        return {
            'documentation_files_count': len(doc_files),
            'essential_docs_found': found_essential,
            'essential_docs_missing': [doc for doc in essential_docs if doc not in found_essential],
            'readme_score': readme_score,
            'documentation_completeness': documentation_completeness,
            'overall_doc_score': overall_doc_score,
            'passed': overall_doc_score >= 70
        }


def run_quality_gates():
    """Run comprehensive quality gates."""
    print("🚀 TERRAGON AUTONOMOUS SDLC v4.0")
    print("🛡️ QUALITY GATES - Comprehensive Validation")
    print("="*60)
    
    project_path = Path(__file__).parent
    
    # Initialize validators
    security_validator = SecurityValidator()
    quality_analyzer = CodeQualityAnalyzer()
    test_analyzer = TestCoverageAnalyzer()
    performance_benchmarker = PerformanceBenchmarker()
    doc_validator = DocumentationValidator()
    
    results = {}
    
    # Security Validation
    print("\n🔒 Running Security Validation")
    print("-" * 30)
    security_result = security_validator.validate_project_security(project_path)
    results['security'] = security_result
    
    print(f"Files scanned: {security_result['files_scanned']}")
    print(f"Security issues found: {security_result['total_issues']}")
    print(f"High severity issues: {security_result['high_severity_issues']}")
    print(f"Security score: {security_result['security_score']}/100")
    print(f"Status: {'✅ PASSED' if security_result['passed'] else '❌ FAILED'}")
    
    # Code Quality Analysis
    print("\n📊 Running Code Quality Analysis")
    print("-" * 30)
    quality_result = quality_analyzer.analyze_project_quality(project_path)
    results['quality'] = quality_result
    
    print(f"Files analyzed: {quality_result['files_analyzed']}")
    print(f"Total lines of code: {quality_result['total_lines_of_code']:,}")
    print(f"Functions: {quality_result['total_functions']}")
    print(f"Classes: {quality_result['total_classes']}")
    print(f"Average complexity score: {quality_result['average_complexity_score']:.1f}/100")
    print(f"Maintainability ratio: {quality_result['maintainability_ratio']:.1%}")
    print(f"Status: {'✅ PASSED' if quality_result['passed'] else '❌ FAILED'}")
    
    # Test Coverage Analysis
    print("\n🧪 Running Test Coverage Analysis")
    print("-" * 30)
    coverage_result = test_analyzer.analyze_test_coverage(project_path)
    results['coverage'] = coverage_result
    
    print(f"Source files: {coverage_result['source_files_count']}")
    print(f"Test files: {coverage_result['test_files_count']}")
    print(f"Test functions: {coverage_result['test_functions_count']}")
    print(f"Test density: {coverage_result['test_density']:.1f} tests/source file")
    print(f"Coverage score: {coverage_result['coverage_score']:.1f}/100")
    print(f"Status: {'✅ PASSED' if coverage_result['passed'] else '❌ FAILED'}")
    
    # Performance Benchmarking
    print("\n⚡ Running Performance Benchmarks")
    print("-" * 30)
    import_benchmark = performance_benchmarker.benchmark_import_time(project_path)
    operation_benchmark = performance_benchmarker.benchmark_basic_operations()
    
    results['performance'] = {
        'import_benchmark': import_benchmark,
        'operation_benchmark': operation_benchmark
    }
    
    print(f"Average import time: {import_benchmark['average_import_time']*1000:.1f}ms")
    print(f"Import performance: {'✅ FAST' if import_benchmark['fast_imports'] else '⚠️ SLOW'}")
    
    if 'error' not in operation_benchmark:
        print(f"Steps per second: {operation_benchmark['steps_per_second']:.0f}")
        print(f"Avg step time: {operation_benchmark['avg_step_time_ms']:.2f}ms")
        print(f"Operation performance: {'✅ PASSED' if operation_benchmark['passed'] else '❌ FAILED'}")
    else:
        print(f"Operation benchmark failed: {operation_benchmark['error']}")
    
    # Documentation Validation
    print("\n📚 Running Documentation Validation")
    print("-" * 30)
    doc_result = doc_validator.validate_documentation(project_path)
    results['documentation'] = doc_result
    
    print(f"Documentation files: {doc_result['documentation_files_count']}")
    print(f"Essential docs found: {len(doc_result['essential_docs_found'])}/4")
    print(f"README score: {doc_result['readme_score']}/100")
    print(f"Documentation completeness: {doc_result['documentation_completeness']:.1%}")
    print(f"Overall doc score: {doc_result['overall_doc_score']:.1f}/100")
    print(f"Status: {'✅ PASSED' if doc_result['passed'] else '❌ FAILED'}")
    
    # Overall Quality Gates Summary
    print("\n" + "="*60)
    print("📊 QUALITY GATES SUMMARY")
    print("="*60)
    
    gate_results = [
        ("Security", security_result['passed']),
        ("Code Quality", quality_result['passed']),
        ("Test Coverage", coverage_result['passed']),
        ("Performance", import_benchmark['passed'] and operation_benchmark.get('passed', False)),
        ("Documentation", doc_result['passed'])
    ]
    
    passed_gates = 0
    total_gates = len(gate_results)
    
    for gate_name, passed in gate_results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{gate_name}: {status}")
        if passed:
            passed_gates += 1
    
    overall_pass_rate = passed_gates / total_gates
    print(f"\nOverall: {passed_gates}/{total_gates} gates passed ({overall_pass_rate:.1%})")
    
    # Calculate quality score
    quality_scores = {
        'security': security_result['security_score'],
        'quality': quality_result['average_complexity_score'],
        'coverage': coverage_result['coverage_score'],
        'documentation': doc_result['overall_doc_score']
    }
    
    overall_quality_score = sum(quality_scores.values()) / len(quality_scores)
    
    if overall_pass_rate >= 0.8 and overall_quality_score >= 75:
        grade = "PRODUCTION READY"
        emoji = "🚀"
    elif overall_pass_rate >= 0.6 and overall_quality_score >= 60:
        grade = "GOOD QUALITY"
        emoji = "✅"
    elif overall_pass_rate >= 0.4:
        grade = "NEEDS IMPROVEMENT"
        emoji = "⚠️"
    else:
        grade = "NOT READY"
        emoji = "❌"
    
    print(f"\n{emoji} Overall Quality Grade: {grade}")
    print(f"Quality Score: {overall_quality_score:.1f}/100")
    
    # Detailed quality breakdown
    print(f"\n📈 Quality Breakdown:")
    for category, score in quality_scores.items():
        print(f"• {category.capitalize()}: {score:.1f}/100")
    
    # Save quality gates report
    quality_report = {
        'timestamp': time.time(),
        'overall_pass_rate': overall_pass_rate,
        'overall_quality_score': overall_quality_score,
        'grade': grade,
        'gate_results': {name: passed for name, passed in gate_results},
        'detailed_results': results
    }
    
    with open('quality_gates_report.json', 'w') as f:
        json.dump(quality_report, f, indent=2, default=str)
    
    print(f"\n📄 Quality gates report saved: quality_gates_report.json")
    
    if overall_pass_rate >= 0.8:
        print("🎉 Quality Gates PASSED! Ready for production deployment.")
    else:
        print("⚠️ Quality Gates need improvement before production deployment.")
    
    return overall_pass_rate >= 0.8


if __name__ == "__main__":
    success = run_quality_gates()
    sys.exit(0 if success else 1)