#!/usr/bin/env python3
"""Comprehensive quality gates validation suite for HF Eco2AI Plugin."""

import os
import json
import time
import subprocess
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
from dataclasses import dataclass


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    name: str
    status: str  # "PASS", "FAIL", "WARN", "SKIP"
    score: float  # 0.0 to 100.0
    details: Dict[str, Any]
    execution_time_ms: float
    critical: bool = False


class SecurityValidator:
    """Advanced security validation and vulnerability scanning."""
    
    def __init__(self):
        self.logger = logging.getLogger("security_validator")
        
    def validate_code_security(self) -> QualityGateResult:
        """Comprehensive code security validation."""
        start_time = time.time()
        
        security_checks = {
            "no_hardcoded_secrets": self._check_hardcoded_secrets(),
            "no_sql_injection": self._check_sql_injection_patterns(),
            "secure_file_operations": self._check_file_operations(),
            "input_validation": self._check_input_validation(),
            "secure_imports": self._check_secure_imports()
        }
        
        total_score = sum(check["score"] for check in security_checks.values())
        avg_score = total_score / len(security_checks)
        
        status = "PASS" if avg_score >= 85 else "WARN" if avg_score >= 70 else "FAIL"
        
        return QualityGateResult(
            name="Code Security",
            status=status,
            score=avg_score,
            details=security_checks,
            execution_time_ms=(time.time() - start_time) * 1000,
            critical=True
        )
    
    def _check_hardcoded_secrets(self) -> Dict[str, Any]:
        """Check for hardcoded secrets and API keys."""
        secret_patterns = [
            r'api[_-]?key["\']?\s*[=:]\s*["\'][^"\']+["\']',
            r'secret[_-]?key["\']?\s*[=:]\s*["\'][^"\']+["\']',
            r'password["\']?\s*[=:]\s*["\'][^"\']+["\']',
            r'token["\']?\s*[=:]\s*["\'][^"\']+["\']',
        ]
        
        violations = []
        python_files = list(Path("src").rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in secret_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        if not self._is_acceptable_secret(match):
                            violations.append({
                                "file": str(file_path),
                                "pattern": pattern,
                                "line": self._find_line_number(content, match)
                            })
                            
            except Exception as e:
                self.logger.warning(f"Could not scan {file_path}: {str(e)}")
        
        score = 100.0 if len(violations) == 0 else max(0, 100 - len(violations) * 20)
        
        return {
            "check": "hardcoded_secrets",
            "violations": violations,
            "violation_count": len(violations),
            "score": score,
            "status": "PASS" if score >= 80 else "FAIL"
        }
    
    def _check_sql_injection_patterns(self) -> Dict[str, Any]:
        """Check for potential SQL injection vulnerabilities."""
        sql_patterns = [
            r'["\'].*\+.*["\'].*sql',
            r'execute\(["\'].*%.*["\']',
            r'query\(["\'].*\+.*["\']',
        ]
        
        violations = []
        python_files = list(Path("src").rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in sql_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    violations.extend([{
                        "file": str(file_path),
                        "pattern": pattern,
                        "match": match
                    } for match in matches])
                            
            except Exception as e:
                self.logger.warning(f"Could not scan {file_path}: {str(e)}")
        
        score = 100.0 if len(violations) == 0 else max(0, 100 - len(violations) * 30)
        
        return {
            "check": "sql_injection",
            "violations": violations,
            "violation_count": len(violations),
            "score": score,
            "status": "PASS" if score >= 80 else "FAIL"
        }
    
    def _check_file_operations(self) -> Dict[str, Any]:
        """Check for secure file operations."""
        unsafe_patterns = [
            r'open\([^)]*["\']w["\']',  # Writing without explicit encoding
            r'pickle\.load\(',  # Unsafe pickle loading
            r'eval\(',  # Dangerous eval usage
            r'exec\(',  # Dangerous exec usage
        ]
        
        violations = []
        python_files = list(Path("src").rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in unsafe_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    violations.extend([{
                        "file": str(file_path),
                        "pattern": pattern,
                        "severity": "HIGH" if "eval" in pattern or "exec" in pattern else "MEDIUM"
                    } for match in matches])
                            
            except Exception as e:
                self.logger.warning(f"Could not scan {file_path}: {str(e)}")
        
        score = 100.0 if len(violations) == 0 else max(0, 100 - len(violations) * 25)
        
        return {
            "check": "file_operations",
            "violations": violations,
            "violation_count": len(violations),
            "score": score,
            "status": "PASS" if score >= 80 else "FAIL"
        }
    
    def _check_input_validation(self) -> Dict[str, Any]:
        """Check for proper input validation patterns."""
        validation_patterns = [
            r'isinstance\(',  # Type checking
            r'len\([^)]+\)\s*[<>]=?\s*\d+',  # Length validation
            r'if.*not.*:',  # Null checks
        ]
        
        total_functions = 0
        validated_functions = 0
        python_files = list(Path("src").rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Count functions
                function_matches = re.findall(r'def\s+\w+\([^)]*\):', content)
                total_functions += len(function_matches)
                
                # Count validation patterns
                for pattern in validation_patterns:
                    if re.search(pattern, content):
                        validated_functions += 1
                        break
                            
            except Exception as e:
                self.logger.warning(f"Could not scan {file_path}: {str(e)}")
        
        validation_ratio = validated_functions / total_functions if total_functions > 0 else 1.0
        score = validation_ratio * 100
        
        return {
            "check": "input_validation",
            "total_functions": total_functions,
            "validated_functions": validated_functions,
            "validation_ratio": validation_ratio,
            "score": score,
            "status": "PASS" if score >= 70 else "WARN" if score >= 50 else "FAIL"
        }
    
    def _check_secure_imports(self) -> Dict[str, Any]:
        """Check for secure import practices."""
        risky_imports = [
            "pickle", "marshal", "shelve", "subprocess", "os.system",
            "eval", "exec", "compile", "__import__"
        ]
        
        violations = []
        python_files = list(Path("src").rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for risky_import in risky_imports:
                    if f"import {risky_import}" in content or f"from {risky_import}" in content:
                        violations.append({
                            "file": str(file_path),
                            "risky_import": risky_import,
                            "severity": "HIGH" if risky_import in ["eval", "exec"] else "MEDIUM"
                        })
                            
            except Exception as e:
                self.logger.warning(f"Could not scan {file_path}: {str(e)}")
        
        score = 100.0 if len(violations) == 0 else max(0, 100 - len(violations) * 15)
        
        return {
            "check": "secure_imports",
            "violations": violations,
            "violation_count": len(violations),
            "score": score,
            "status": "PASS" if score >= 80 else "WARN"
        }
    
    def _is_acceptable_secret(self, secret_text: str) -> bool:
        """Check if a detected secret is acceptable (test data, placeholders, etc.)."""
        acceptable_patterns = [
            "your_api_key", "example", "placeholder", "test", "dummy", 
            "sample", "mock", "fake", "***", "REDACTED"
        ]
        
        return any(pattern.lower() in secret_text.lower() for pattern in acceptable_patterns)
    
    def _find_line_number(self, content: str, search_text: str) -> int:
        """Find line number of text in content."""
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if search_text in line:
                return i
        return 0


class PerformanceBenchmarker:
    """Performance benchmarking and optimization validation."""
    
    def __init__(self):
        self.logger = logging.getLogger("performance_benchmarker")
        
    def validate_performance(self) -> QualityGateResult:
        """Comprehensive performance validation."""
        start_time = time.time()
        
        performance_checks = {
            "code_efficiency": self._check_code_efficiency(),
            "memory_usage": self._check_memory_patterns(),
            "computational_complexity": self._check_algorithmic_complexity(),
            "resource_optimization": self._check_resource_optimization()
        }
        
        total_score = sum(check["score"] for check in performance_checks.values())
        avg_score = total_score / len(performance_checks)
        
        status = "PASS" if avg_score >= 80 else "WARN" if avg_score >= 60 else "FAIL"
        
        return QualityGateResult(
            name="Performance",
            status=status,
            score=avg_score,
            details=performance_checks,
            execution_time_ms=(time.time() - start_time) * 1000
        )
    
    def _check_code_efficiency(self) -> Dict[str, Any]:
        """Check for code efficiency patterns."""
        efficiency_violations = []
        python_files = list(Path("src").rglob("*.py"))
        
        # Patterns that indicate inefficient code
        inefficient_patterns = [
            r'for.*in.*range\(len\([^)]+\)\):',  # Using range(len()) instead of enumerate
            r'\.append\([^)]+\)\s*for.*in',  # List comprehension would be better
            r'\+\s*=.*\+',  # String concatenation in loop
        ]
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in inefficient_patterns:
                    matches = re.findall(pattern, content)
                    efficiency_violations.extend([{
                        "file": str(file_path),
                        "pattern": pattern,
                        "improvement": self._get_efficiency_suggestion(pattern)
                    } for match in matches])
                            
            except Exception as e:
                self.logger.warning(f"Could not scan {file_path}: {str(e)}")
        
        score = max(0, 100 - len(efficiency_violations) * 10)
        
        return {
            "check": "code_efficiency",
            "violations": efficiency_violations,
            "violation_count": len(efficiency_violations),
            "score": score,
            "status": "PASS" if score >= 80 else "WARN"
        }
    
    def _check_memory_patterns(self) -> Dict[str, Any]:
        """Check for memory-efficient patterns."""
        memory_issues = []
        python_files = list(Path("src").rglob("*.py"))
        
        # Patterns that may cause memory issues
        memory_patterns = [
            r'\.append\([^)]+\).*for.*in.*range\(\d{4,}\)',  # Large list creation
            r'open\([^)]*\)\.read\(\)',  # Reading entire file without context manager
            r'\[\].*for.*in.*\[.*\]',  # Nested list comprehensions
        ]
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in memory_patterns:
                    matches = re.findall(pattern, content)
                    memory_issues.extend([{
                        "file": str(file_path),
                        "pattern": pattern,
                        "severity": "MEDIUM"
                    } for match in matches])
                            
            except Exception as e:
                self.logger.warning(f"Could not scan {file_path}: {str(e)}")
        
        score = max(0, 100 - len(memory_issues) * 15)
        
        return {
            "check": "memory_patterns",
            "issues": memory_issues,
            "issue_count": len(memory_issues),
            "score": score,
            "status": "PASS" if score >= 80 else "WARN"
        }
    
    def _check_algorithmic_complexity(self) -> Dict[str, Any]:
        """Analyze algorithmic complexity patterns."""
        complexity_issues = []
        python_files = list(Path("src").rglob("*.py"))
        
        # Patterns that indicate high complexity
        complexity_patterns = [
            (r'for.*for.*for', "O(n³)", "HIGH"),  # Triple nested loops
            (r'for.*for.*in', "O(n²)", "MEDIUM"),  # Nested loops
            (r'while.*while', "Potentially infinite", "HIGH"),  # Nested while loops
        ]
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern, complexity, severity in complexity_patterns:
                    matches = re.findall(pattern, content)
                    complexity_issues.extend([{
                        "file": str(file_path),
                        "complexity": complexity,
                        "severity": severity,
                        "count": len(matches)
                    } for match in matches])
                            
            except Exception as e:
                self.logger.warning(f"Could not scan {file_path}: {str(e)}")
        
        high_complexity_count = sum(1 for issue in complexity_issues if issue["severity"] == "HIGH")
        score = max(0, 100 - high_complexity_count * 30 - len(complexity_issues) * 5)
        
        return {
            "check": "algorithmic_complexity",
            "issues": complexity_issues,
            "high_complexity_count": high_complexity_count,
            "total_issues": len(complexity_issues),
            "score": score,
            "status": "PASS" if score >= 70 else "WARN"
        }
    
    def _check_resource_optimization(self) -> Dict[str, Any]:
        """Check for resource optimization patterns."""
        optimization_patterns = [
            r'@lru_cache',  # Caching decorators
            r'with\s+open',  # Context managers
            r'asyncio\.',  # Async operations
            r'threading\.',  # Threading usage
            r'multiprocessing\.',  # Multiprocessing usage
        ]
        
        optimization_count = 0
        total_files = 0
        python_files = list(Path("src").rglob("*.py"))
        
        for file_path in python_files:
            total_files += 1
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in optimization_patterns:
                    if re.search(pattern, content):
                        optimization_count += 1
                        break
                            
            except Exception as e:
                self.logger.warning(f"Could not scan {file_path}: {str(e)}")
        
        optimization_ratio = optimization_count / total_files if total_files > 0 else 0
        score = optimization_ratio * 100
        
        return {
            "check": "resource_optimization",
            "optimized_files": optimization_count,
            "total_files": total_files,
            "optimization_ratio": optimization_ratio,
            "score": score,
            "status": "PASS" if score >= 60 else "WARN"
        }
    
    def _get_efficiency_suggestion(self, pattern: str) -> str:
        """Get efficiency improvement suggestion for a pattern."""
        suggestions = {
            r'for.*in.*range\(len\([^)]+\)\):': "Use enumerate() instead of range(len())",
            r'\.append\([^)]+\)\s*for.*in': "Consider using list comprehension",
            r'\+\s*=.*\+': "Use join() for string concatenation in loops"
        }
        
        for pattern_key, suggestion in suggestions.items():
            if pattern_key in pattern:
                return suggestion
                
        return "Consider optimizing this pattern"


class ComprehensiveQualityGateValidator:
    """Main orchestrator for all quality gate validations."""
    
    def __init__(self):
        self.security_validator = SecurityValidator()
        self.performance_benchmarker = PerformanceBenchmarker()
        self.logger = logging.getLogger("quality_gates")
        
    def run_all_quality_gates(self) -> Dict[str, Any]:
        """Run all quality gates and generate comprehensive report."""
        start_time = time.time()
        
        self.logger.info("🔍 Running comprehensive quality gate validation...")
        
        # Execute all quality gates
        quality_gates = {
            "security": self.security_validator.validate_code_security(),
            "performance": self.performance_benchmarker.validate_performance(),
            "code_structure": self._validate_code_structure(),
            "documentation": self._validate_documentation(),
            "testing": self._validate_testing_coverage()
        }
        
        # Calculate overall scores
        total_score = sum(gate.score for gate in quality_gates.values())
        avg_score = total_score / len(quality_gates)
        
        # Determine overall status
        critical_failures = [name for name, gate in quality_gates.items() 
                           if gate.critical and gate.status == "FAIL"]
        
        if critical_failures:
            overall_status = "CRITICAL_FAIL"
        elif avg_score >= 85:
            overall_status = "EXCELLENT"
        elif avg_score >= 75:
            overall_status = "GOOD"
        elif avg_score >= 60:
            overall_status = "ACCEPTABLE"
        else:
            overall_status = "NEEDS_IMPROVEMENT"
        
        total_time = time.time() - start_time
        
        return {
            "timestamp": time.time(),
            "overall_status": overall_status,
            "overall_score": avg_score,
            "execution_time_seconds": total_time,
            "critical_failures": critical_failures,
            "quality_gates": {name: {
                "status": gate.status,
                "score": gate.score,
                "execution_time_ms": gate.execution_time_ms,
                "critical": gate.critical,
                "details": gate.details
            } for name, gate in quality_gates.items()},
            "recommendations": self._generate_recommendations(quality_gates)
        }
    
    def _validate_code_structure(self) -> QualityGateResult:
        """Validate code structure and organization."""
        start_time = time.time()
        
        structure_checks = {
            "module_organization": self._check_module_organization(),
            "import_structure": self._check_import_structure(),
            "class_design": self._check_class_design(),
            "function_complexity": self._check_function_complexity()
        }
        
        total_score = sum(check["score"] for check in structure_checks.values())
        avg_score = total_score / len(structure_checks)
        
        status = "PASS" if avg_score >= 80 else "WARN" if avg_score >= 60 else "FAIL"
        
        return QualityGateResult(
            name="Code Structure",
            status=status,
            score=avg_score,
            details=structure_checks,
            execution_time_ms=(time.time() - start_time) * 1000
        )
    
    def _validate_documentation(self) -> QualityGateResult:
        """Validate documentation coverage and quality."""
        start_time = time.time()
        
        doc_files = [
            "README.md", "CONTRIBUTING.md", "SECURITY.md", 
            "CHANGELOG.md", "LICENSE"
        ]
        
        existing_docs = sum(1 for doc in doc_files if Path(doc).exists())
        doc_coverage = existing_docs / len(doc_files)
        
        # Check for docstrings in Python files
        python_files = list(Path("src").rglob("*.py"))
        documented_functions = 0
        total_functions = 0
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Count functions and docstrings
                functions = re.findall(r'def\s+\w+\([^)]*\):', content)
                docstrings = re.findall(r'"""[^"]*"""', content) + re.findall(r"'''[^']*'''", content)
                
                total_functions += len(functions)
                documented_functions += min(len(functions), len(docstrings))
                        
            except Exception as e:
                self.logger.warning(f"Could not scan {file_path}: {str(e)}")
        
        docstring_coverage = documented_functions / total_functions if total_functions > 0 else 1.0
        
        overall_score = (doc_coverage * 40) + (docstring_coverage * 60)
        status = "PASS" if overall_score >= 70 else "WARN" if overall_score >= 50 else "FAIL"
        
        return QualityGateResult(
            name="Documentation",
            status=status,
            score=overall_score,
            details={
                "documentation_files": {
                    "existing": existing_docs,
                    "total": len(doc_files),
                    "coverage": doc_coverage
                },
                "docstring_coverage": {
                    "documented_functions": documented_functions,
                    "total_functions": total_functions,
                    "coverage": docstring_coverage
                }
            },
            execution_time_ms=(time.time() - start_time) * 1000
        )
    
    def _validate_testing_coverage(self) -> QualityGateResult:
        """Validate testing infrastructure and coverage."""
        start_time = time.time()
        
        test_files = list(Path("tests").rglob("test_*.py")) + list(Path(".").glob("test_*.py"))
        src_files = list(Path("src").rglob("*.py"))
        
        test_coverage_ratio = len(test_files) / len(src_files) if src_files else 0
        
        # Check for different types of tests
        test_types = {
            "unit": len(list(Path("tests/unit").rglob("*.py"))) if Path("tests/unit").exists() else 0,
            "integration": len(list(Path("tests/integration").rglob("*.py"))) if Path("tests/integration").exists() else 0,
            "e2e": len(list(Path("tests/e2e").rglob("*.py"))) if Path("tests/e2e").exists() else 0,
            "performance": len(list(Path("tests/performance").rglob("*.py"))) if Path("tests/performance").exists() else 0
        }
        
        test_diversity_score = sum(1 for count in test_types.values() if count > 0) / len(test_types) * 100
        
        overall_score = (test_coverage_ratio * 60) + (test_diversity_score * 40)
        status = "PASS" if overall_score >= 70 else "WARN" if overall_score >= 50 else "FAIL"
        
        return QualityGateResult(
            name="Testing",
            status=status,
            score=overall_score,
            details={
                "test_coverage_ratio": test_coverage_ratio,
                "test_files": len(test_files),
                "src_files": len(src_files),
                "test_types": test_types,
                "test_diversity_score": test_diversity_score
            },
            execution_time_ms=(time.time() - start_time) * 1000
        )
    
    def _check_module_organization(self) -> Dict[str, Any]:
        """Check module organization patterns."""
        src_path = Path("src")
        if not src_path.exists():
            return {"score": 0, "status": "FAIL", "reason": "No src directory found"}
        
        # Check for proper package structure
        init_files = list(src_path.rglob("__init__.py"))
        python_dirs = len([p for p in src_path.rglob("*") if p.is_dir()])
        
        organization_score = min(100, len(init_files) / max(1, python_dirs) * 100)
        
        return {
            "score": organization_score,
            "init_files": len(init_files),
            "python_directories": python_dirs,
            "status": "PASS" if organization_score >= 80 else "WARN"
        }
    
    def _check_import_structure(self) -> Dict[str, Any]:
        """Check import structure and organization."""
        import_violations = []
        python_files = list(Path("src").rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                import_section_ended = False
                for i, line in enumerate(lines):
                    stripped_line = line.strip()
                    
                    if stripped_line and not stripped_line.startswith('#'):
                        if stripped_line.startswith(('import ', 'from ')):
                            if import_section_ended:
                                import_violations.append({
                                    "file": str(file_path),
                                    "line": i + 1,
                                    "issue": "Import not at top of file"
                                })
                        else:
                            import_section_ended = True
                            
            except Exception as e:
                self.logger.warning(f"Could not scan {file_path}: {str(e)}")
        
        score = max(0, 100 - len(import_violations) * 10)
        
        return {
            "score": score,
            "violations": import_violations,
            "violation_count": len(import_violations),
            "status": "PASS" if score >= 80 else "WARN"
        }
    
    def _check_class_design(self) -> Dict[str, Any]:
        """Check class design patterns."""
        design_issues = []
        python_files = list(Path("src").rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for very large classes (>500 lines)
                classes = re.findall(r'class\s+\w+[^:]*:', content)
                for class_match in classes:
                    # Simple heuristic: count methods in class
                    class_start = content.find(class_match)
                    class_section = content[class_start:class_start + 10000]  # Look ahead 10k chars
                    methods = re.findall(r'\n\s+def\s+', class_section)
                    
                    if len(methods) > 20:  # Large class
                        design_issues.append({
                            "file": str(file_path),
                            "issue": f"Large class with {len(methods)} methods",
                            "severity": "MEDIUM"
                        })
                            
            except Exception as e:
                self.logger.warning(f"Could not scan {file_path}: {str(e)}")
        
        score = max(0, 100 - len(design_issues) * 15)
        
        return {
            "score": score,
            "issues": design_issues,
            "issue_count": len(design_issues),
            "status": "PASS" if score >= 80 else "WARN"
        }
    
    def _check_function_complexity(self) -> Dict[str, Any]:
        """Check function complexity using cyclomatic complexity approximation."""
        complexity_issues = []
        python_files = list(Path("src").rglob("*.py"))
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                functions = re.findall(r'def\s+(\w+)\([^)]*\):[^}]*', content)
                
                for function_name in functions:
                    # Simple complexity estimation based on control structures
                    function_pattern = rf'def\s+{function_name}\([^)]*\):.*?(?=\ndef|\nclass|\Z)'
                    function_match = re.search(function_pattern, content, re.DOTALL)
                    
                    if function_match:
                        function_body = function_match.group(0)
                        
                        # Count complexity indicators
                        complexity = 1  # Base complexity
                        complexity += len(re.findall(r'\bif\b', function_body))
                        complexity += len(re.findall(r'\bfor\b', function_body))
                        complexity += len(re.findall(r'\bwhile\b', function_body))
                        complexity += len(re.findall(r'\btry\b', function_body))
                        complexity += len(re.findall(r'\bexcept\b', function_body))
                        
                        if complexity > 10:  # High complexity threshold
                            complexity_issues.append({
                                "file": str(file_path),
                                "function": function_name,
                                "complexity": complexity,
                                "severity": "HIGH" if complexity > 15 else "MEDIUM"
                            })
                            
            except Exception as e:
                self.logger.warning(f"Could not scan {file_path}: {str(e)}")
        
        high_complexity_count = sum(1 for issue in complexity_issues if issue["severity"] == "HIGH")
        score = max(0, 100 - high_complexity_count * 25 - len(complexity_issues) * 10)
        
        return {
            "score": score,
            "issues": complexity_issues,
            "high_complexity_count": high_complexity_count,
            "total_issues": len(complexity_issues),
            "status": "PASS" if score >= 70 else "WARN"
        }
    
    def _generate_recommendations(self, quality_gates: Dict[str, QualityGateResult]) -> List[Dict[str, str]]:
        """Generate recommendations based on quality gate results."""
        recommendations = []
        
        for name, gate in quality_gates.items():
            if gate.status == "FAIL":
                recommendations.append({
                    "priority": "HIGH" if gate.critical else "MEDIUM",
                    "category": name,
                    "recommendation": f"Address {name} issues - score: {gate.score:.1f}",
                    "action": "Review detailed findings and implement fixes"
                })
            elif gate.status == "WARN":
                recommendations.append({
                    "priority": "LOW",
                    "category": name,
                    "recommendation": f"Improve {name} quality - score: {gate.score:.1f}",
                    "action": "Consider enhancements for better quality"
                })
        
        # Add general recommendations
        recommendations.append({
            "priority": "LOW",
            "category": "Continuous Improvement",
            "recommendation": "Implement automated quality gates in CI/CD pipeline",
            "action": "Set up automated quality checks on pull requests"
        })
        
        return recommendations


async def main():
    """Run comprehensive quality gates validation."""
    print("🔍 HF Eco2AI Comprehensive Quality Gates Validation")
    print("=" * 55)
    
    # Initialize validator
    validator = ComprehensiveQualityGateValidator()
    
    # Run all quality gates
    print("\n🧪 Running all quality gate validations...")
    results = validator.run_all_quality_gates()
    
    # Display results
    print(f"\n📊 QUALITY GATES RESULTS")
    print("=" * 25)
    print(f"Overall Status: {results['overall_status']}")
    print(f"Overall Score: {results['overall_score']:.1f}/100")
    print(f"Execution Time: {results['execution_time_seconds']:.2f}s")
    
    if results['critical_failures']:
        print(f"❌ Critical Failures: {', '.join(results['critical_failures'])}")
    
    print(f"\n🎯 INDIVIDUAL GATE SCORES")
    for name, gate in results['quality_gates'].items():
        status_icon = "✅" if gate['status'] == "PASS" else "⚠️" if gate['status'] == "WARN" else "❌"
        critical_marker = " [CRITICAL]" if gate['critical'] else ""
        print(f"{status_icon} {name.title()}: {gate['score']:.1f}/100{critical_marker}")
    
    print(f"\n💡 TOP RECOMMENDATIONS")
    for rec in results['recommendations'][:3]:  # Show top 3
        priority_icon = "🔴" if rec['priority'] == "HIGH" else "🟡" if rec['priority'] == "MEDIUM" else "🔵"
        print(f"{priority_icon} {rec['recommendation']}")
    
    # Save detailed results
    results_path = "/root/repo/quality_gates_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Detailed results saved to: {results_path}")
    print("✅ Quality gates validation completed!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())