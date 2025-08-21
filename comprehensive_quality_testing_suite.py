"""
🛡️ Comprehensive Quality Testing Suite
Advanced testing framework for autonomous SDLC validation
"""

import asyncio
import logging
import json
import sys
import subprocess
import importlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class QualityTestResult:
    """Quality test result"""
    test_category: str
    test_name: str
    passed: bool
    score: float
    message: str
    execution_time: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class ComprehensiveQualityTester:
    """Comprehensive quality testing framework"""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.results = []
        self.test_suite_start = datetime.now()
    
    async def test_python_environment(self) -> List[QualityTestResult]:
        """Test Python environment and dependencies"""
        logger.info("🐍 Testing Python environment...")
        
        results = []
        start_time = datetime.now()
        
        # Test Python version
        try:
            python_version = sys.version_info
            passed = python_version >= (3, 10)
            score = 100.0 if passed else 50.0
            
            results.append(QualityTestResult(
                test_category="environment",
                test_name="python_version",
                passed=passed,
                score=score,
                message=f"Python {python_version.major}.{python_version.minor}.{python_version.micro}",
                execution_time=(datetime.now() - start_time).total_seconds()
            ))
            
        except Exception as e:
            results.append(QualityTestResult(
                test_category="environment",
                test_name="python_version",
                passed=False,
                score=0.0,
                message=f"Python version check failed: {e}",
                execution_time=(datetime.now() - start_time).total_seconds()
            ))
        
        # Test core imports
        core_imports = ['json', 'asyncio', 'logging', 'pathlib', 'datetime']
        for import_name in core_imports:
            start_time = datetime.now()
            try:
                importlib.import_module(import_name)
                results.append(QualityTestResult(
                    test_category="environment",
                    test_name=f"import_{import_name}",
                    passed=True,
                    score=100.0,
                    message=f"Successfully imported {import_name}",
                    execution_time=(datetime.now() - start_time).total_seconds()
                ))
            except ImportError as e:
                results.append(QualityTestResult(
                    test_category="environment",
                    test_name=f"import_{import_name}",
                    passed=False,
                    score=0.0,
                    message=f"Failed to import {import_name}: {e}",
                    execution_time=(datetime.now() - start_time).total_seconds()
                ))
        
        return results
    
    async def test_code_structure(self) -> List[QualityTestResult]:
        """Test repository code structure"""
        logger.info("🏗️ Testing code structure...")
        
        results = []
        
        # Test directory structure
        required_dirs = [
            "src/hf_eco2ai",
            "tests",
            "docs",
            "deployment"
        ]
        
        for dir_path in required_dirs:
            start_time = datetime.now()
            full_path = self.repo_path / dir_path
            
            if full_path.exists() and full_path.is_dir():
                results.append(QualityTestResult(
                    test_category="structure",
                    test_name=f"directory_{dir_path.replace('/', '_')}",
                    passed=True,
                    score=100.0,
                    message=f"Directory {dir_path} exists",
                    execution_time=(datetime.now() - start_time).total_seconds()
                ))
            else:
                results.append(QualityTestResult(
                    test_category="structure",
                    test_name=f"directory_{dir_path.replace('/', '_')}",
                    passed=False,
                    score=0.0,
                    message=f"Directory {dir_path} missing",
                    execution_time=(datetime.now() - start_time).total_seconds()
                ))
        
        # Test required files
        required_files = [
            "pyproject.toml",
            "README.md",
            "src/hf_eco2ai/__init__.py"
        ]
        
        for file_path in required_files:
            start_time = datetime.now()
            full_path = self.repo_path / file_path
            
            if full_path.exists() and full_path.is_file():
                results.append(QualityTestResult(
                    test_category="structure",
                    test_name=f"file_{file_path.replace('/', '_').replace('.', '_')}",
                    passed=True,
                    score=100.0,
                    message=f"File {file_path} exists",
                    execution_time=(datetime.now() - start_time).total_seconds()
                ))
            else:
                results.append(QualityTestResult(
                    test_category="structure",
                    test_name=f"file_{file_path.replace('/', '_').replace('.', '_')}",
                    passed=False,
                    score=0.0,
                    message=f"File {file_path} missing",
                    execution_time=(datetime.now() - start_time).total_seconds()
                ))
        
        return results
    
    async def test_code_quality(self) -> List[QualityTestResult]:
        """Test code quality metrics"""
        logger.info("✨ Testing code quality...")
        
        results = []
        
        # Count Python files
        start_time = datetime.now()
        try:
            python_files = list(self.repo_path.glob("**/*.py"))
            file_count = len(python_files)
            
            # Quality score based on file count
            if file_count >= 50:
                score = 100.0
                message = f"Excellent: {file_count} Python files"
            elif file_count >= 20:
                score = 80.0
                message = f"Good: {file_count} Python files"
            elif file_count >= 5:
                score = 60.0
                message = f"Adequate: {file_count} Python files"
            else:
                score = 30.0
                message = f"Minimal: {file_count} Python files"
            
            results.append(QualityTestResult(
                test_category="quality",
                test_name="python_file_count",
                passed=file_count >= 5,
                score=score,
                message=message,
                execution_time=(datetime.now() - start_time).total_seconds()
            ))
            
        except Exception as e:
            results.append(QualityTestResult(
                test_category="quality",
                test_name="python_file_count",
                passed=False,
                score=0.0,
                message=f"Failed to count Python files: {e}",
                execution_time=(datetime.now() - start_time).total_seconds()
            ))
        
        # Test configuration quality
        start_time = datetime.now()
        try:
            pyproject_path = self.repo_path / "pyproject.toml"
            if pyproject_path.exists():
                with open(pyproject_path, 'r') as f:
                    content = f.read()
                
                # Check for key configuration sections
                config_checks = [
                    ('build-system' in content, "Build system configured"),
                    ('dependencies' in content, "Dependencies defined"),
                    ('optional-dependencies' in content, "Optional dependencies defined"),
                    ('project' in content, "Project metadata defined")
                ]
                
                passed_checks = sum(1 for check, _ in config_checks if check)
                total_checks = len(config_checks)
                score = (passed_checks / total_checks) * 100
                
                results.append(QualityTestResult(
                    test_category="quality",
                    test_name="configuration_completeness",
                    passed=passed_checks >= 3,
                    score=score,
                    message=f"Configuration: {passed_checks}/{total_checks} sections complete",
                    execution_time=(datetime.now() - start_time).total_seconds()
                ))
            else:
                results.append(QualityTestResult(
                    test_category="quality",
                    test_name="configuration_completeness",
                    passed=False,
                    score=0.0,
                    message="pyproject.toml not found",
                    execution_time=(datetime.now() - start_time).total_seconds()
                ))
                
        except Exception as e:
            results.append(QualityTestResult(
                test_category="quality",
                test_name="configuration_completeness",
                passed=False,
                score=0.0,
                message=f"Configuration check failed: {e}",
                execution_time=(datetime.now() - start_time).total_seconds()
            ))
        
        return results
    
    async def test_functionality(self) -> List[QualityTestResult]:
        """Test core functionality"""
        logger.info("⚙️ Testing functionality...")
        
        results = []
        
        # Test autonomous validator
        start_time = datetime.now()
        try:
            validator_path = self.repo_path / "autonomous_validator.py"
            if validator_path.exists():
                result = subprocess.run([
                    sys.executable, str(validator_path)
                ], capture_output=True, text=True, cwd=str(self.repo_path), timeout=30)
                
                if result.returncode == 0:
                    results.append(QualityTestResult(
                        test_category="functionality",
                        test_name="autonomous_validator",
                        passed=True,
                        score=100.0,
                        message="Autonomous validator working correctly",
                        execution_time=(datetime.now() - start_time).total_seconds()
                    ))
                else:
                    results.append(QualityTestResult(
                        test_category="functionality",
                        test_name="autonomous_validator",
                        passed=False,
                        score=25.0,
                        message=f"Validator issues: {result.stderr[:100]}",
                        execution_time=(datetime.now() - start_time).total_seconds()
                    ))
            else:
                results.append(QualityTestResult(
                    test_category="functionality",
                    test_name="autonomous_validator",
                    passed=False,
                    score=0.0,
                    message="Autonomous validator not found",
                    execution_time=(datetime.now() - start_time).total_seconds()
                ))
                
        except Exception as e:
            results.append(QualityTestResult(
                test_category="functionality",
                test_name="autonomous_validator",
                passed=False,
                score=0.0,
                message=f"Validator test failed: {e}",
                execution_time=(datetime.now() - start_time).total_seconds()
            ))
        
        # Test enhanced demo
        start_time = datetime.now()
        demo_path = self.repo_path / "enhanced_carbon_intelligence_demo.py"
        if demo_path.exists():
            results.append(QualityTestResult(
                test_category="functionality",
                test_name="enhanced_demo_exists",
                passed=True,
                score=100.0,
                message="Enhanced demo file exists",
                execution_time=(datetime.now() - start_time).total_seconds()
            ))
        else:
            results.append(QualityTestResult(
                test_category="functionality",
                test_name="enhanced_demo_exists",
                passed=False,
                score=0.0,
                message="Enhanced demo file missing",
                execution_time=(datetime.now() - start_time).total_seconds()
            ))
        
        return results
    
    async def test_security(self) -> List[QualityTestResult]:
        """Test security measures"""
        logger.info("🔒 Testing security...")
        
        results = []
        
        # Check for security-related files
        security_files = [
            "SECURITY.md",
            ".bandit",
            "scripts/security-scan.sh"
        ]
        
        for file_path in security_files:
            start_time = datetime.now()
            full_path = self.repo_path / file_path
            
            if full_path.exists():
                results.append(QualityTestResult(
                    test_category="security",
                    test_name=f"security_file_{file_path.replace('/', '_').replace('.', '_')}",
                    passed=True,
                    score=100.0,
                    message=f"Security file {file_path} exists",
                    execution_time=(datetime.now() - start_time).total_seconds()
                ))
            else:
                results.append(QualityTestResult(
                    test_category="security",
                    test_name=f"security_file_{file_path.replace('/', '_').replace('.', '_')}",
                    passed=False,
                    score=50.0,  # Not critical but good to have
                    message=f"Security file {file_path} missing",
                    execution_time=(datetime.now() - start_time).total_seconds()
                ))
        
        return results
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all quality tests"""
        logger.info("🚀 Starting comprehensive quality testing...")
        
        # Run all test categories
        test_categories = [
            ("environment", self.test_python_environment()),
            ("structure", self.test_code_structure()),
            ("quality", self.test_code_quality()),
            ("functionality", self.test_functionality()),
            ("security", self.test_security())
        ]
        
        all_results = []
        category_summaries = {}
        
        for category_name, test_coroutine in test_categories:
            logger.info(f"Running {category_name} tests...")
            try:
                category_results = await test_coroutine
                all_results.extend(category_results)
                
                # Calculate category summary
                category_tests = [r for r in category_results if r.test_category == category_name]
                total_tests = len(category_tests)
                passed_tests = sum(1 for r in category_tests if r.passed)
                avg_score = sum(r.score for r in category_tests) / total_tests if total_tests > 0 else 0.0
                
                category_summaries[category_name] = {
                    'total_tests': total_tests,
                    'passed_tests': passed_tests,
                    'failed_tests': total_tests - passed_tests,
                    'pass_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0.0,
                    'average_score': avg_score
                }
                
            except Exception as e:
                logger.error(f"Error running {category_name} tests: {e}")
                category_summaries[category_name] = {
                    'total_tests': 0,
                    'passed_tests': 0,
                    'failed_tests': 1,
                    'pass_rate': 0.0,
                    'average_score': 0.0,
                    'error': str(e)
                }
        
        # Calculate overall summary
        total_tests = len(all_results)
        passed_tests = sum(1 for r in all_results if r.passed)
        overall_score = sum(r.score for r in all_results) / total_tests if total_tests > 0 else 0.0
        
        test_duration = (datetime.now() - self.test_suite_start).total_seconds()
        
        summary = {
            'test_suite_summary': {
                'start_time': self.test_suite_start.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': test_duration,
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'overall_pass_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0.0,
                'overall_score': overall_score,
                'quality_grade': self._calculate_quality_grade(overall_score)
            },
            'category_summaries': category_summaries,
            'detailed_results': [
                {
                    'category': r.test_category,
                    'test_name': r.test_name,
                    'passed': r.passed,
                    'score': r.score,
                    'message': r.message,
                    'execution_time': r.execution_time
                } for r in all_results
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        results_path = self.repo_path / "comprehensive_quality_report.json"
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📊 Quality testing completed:")
        logger.info(f"   Total tests: {total_tests}")
        logger.info(f"   Passed: {passed_tests}")
        logger.info(f"   Failed: {total_tests - passed_tests}")
        logger.info(f"   Overall score: {overall_score:.1f}/100")
        logger.info(f"   Quality grade: {summary['test_suite_summary']['quality_grade']}")
        logger.info(f"   Results saved to: {results_path}")
        
        return summary
    
    def _calculate_quality_grade(self, score: float) -> str:
        """Calculate quality grade based on score"""
        if score >= 90:
            return "A+ (Excellent)"
        elif score >= 80:
            return "A (Very Good)"
        elif score >= 70:
            return "B+ (Good)"
        elif score >= 60:
            return "B (Satisfactory)"
        elif score >= 50:
            return "C (Needs Improvement)"
        else:
            return "D (Poor)"

async def main():
    """Main quality testing entry point"""
    tester = ComprehensiveQualityTester(Path("/root/repo"))
    
    print("🛡️ Comprehensive Quality Testing Suite")
    print("=" * 50)
    
    results = await tester.run_comprehensive_tests()
    
    print(f"\n📊 Quality Test Results:")
    print(f"Total Tests: {results['test_suite_summary']['total_tests']}")
    print(f"Passed: {results['test_suite_summary']['passed_tests']}")
    print(f"Failed: {results['test_suite_summary']['failed_tests']}")
    print(f"Pass Rate: {results['test_suite_summary']['overall_pass_rate']:.1f}%")
    print(f"Overall Score: {results['test_suite_summary']['overall_score']:.1f}/100")
    print(f"Quality Grade: {results['test_suite_summary']['quality_grade']}")
    
    return results['test_suite_summary']['overall_score'] >= 70.0

if __name__ == "__main__":
    asyncio.run(main())