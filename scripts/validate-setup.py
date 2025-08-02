#!/usr/bin/env python3
"""Comprehensive setup validation script for HF Eco2AI Plugin."""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
import xml.etree.ElementTree as ET

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SetupValidator:
    """Validate complete project setup and configuration."""
    
    def __init__(self):
        self.repo_root = Path.cwd()
        self.validation_results = {
            'overall_status': 'unknown',
            'categories': {},
            'recommendations': [],
            'critical_issues': [],
            'warnings': []
        }
    
    def validate_project_structure(self) -> Dict[str, Any]:
        """Validate project directory structure."""
        logger.info("Validating project structure...")
        
        required_files = [
            'pyproject.toml',
            'README.md',
            'LICENSE',
            '.gitignore',
            '.pre-commit-config.yaml',
            'src/hf_eco2ai/__init__.py',
            'tests/__init__.py'
        ]
        
        required_dirs = [
            'src',
            'tests',
            'docs',
            'scripts',
            '.github'
        ]
        
        structure_report = {
            'status': 'pass',
            'missing_files': [],
            'missing_dirs': [],
            'optional_missing': []
        }
        
        # Check required files
        for file_path in required_files:
            if not (self.repo_root / file_path).exists():
                structure_report['missing_files'].append(file_path)
                structure_report['status'] = 'fail'
        
        # Check required directories
        for dir_path in required_dirs:
            if not (self.repo_root / dir_path).is_dir():
                structure_report['missing_dirs'].append(dir_path)
                structure_report['status'] = 'fail'
        
        # Check optional files
        optional_files = [
            'CHANGELOG.md',
            'CONTRIBUTING.md',
            'CODE_OF_CONDUCT.md',
            'SECURITY.md',
            'Dockerfile',
            'docker-compose.yml'
        ]
        
        for file_path in optional_files:
            if not (self.repo_root / file_path).exists():
                structure_report['optional_missing'].append(file_path)
        
        return structure_report
    
    def validate_python_environment(self) -> Dict[str, Any]:
        """Validate Python environment and dependencies."""
        logger.info("Validating Python environment...")
        
        env_report = {
            'status': 'pass',
            'python_version': None,
            'pip_version': None,
            'dependencies_installed': False,
            'dev_dependencies_installed': False,
            'issues': []
        }
        
        try:
            # Check Python version
            result = subprocess.run([sys.executable, '--version'], 
                                  capture_output=True, text=True, check=True)
            env_report['python_version'] = result.stdout.strip()
            
            # Check pip version
            result = subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                                  capture_output=True, text=True, check=True)
            env_report['pip_version'] = result.stdout.strip()
            
            # Try to import the main package
            try:
                import hf_eco2ai
                env_report['dependencies_installed'] = True
            except ImportError:
                env_report['dependencies_installed'] = False
                env_report['issues'].append('Main package not importable')
                env_report['status'] = 'warning'
            
            # Check development dependencies
            dev_tools = ['pytest', 'black', 'flake8', 'mypy', 'pre-commit']
            missing_dev_tools = []
            
            for tool in dev_tools:
                try:
                    subprocess.run([sys.executable, '-m', tool, '--version'], 
                                 capture_output=True, check=True)
                except subprocess.CalledProcessError:
                    missing_dev_tools.append(tool)
            
            if missing_dev_tools:
                env_report['dev_dependencies_installed'] = False
                env_report['issues'].append(f'Missing dev tools: {", ".join(missing_dev_tools)}')
                env_report['status'] = 'warning'
            else:
                env_report['dev_dependencies_installed'] = True
        
        except subprocess.CalledProcessError as e:
            env_report['status'] = 'fail'
            env_report['issues'].append(f'Environment check failed: {e}')
        
        return env_report
    
    def validate_configuration_files(self) -> Dict[str, Any]:
        """Validate configuration files syntax and content."""
        logger.info("Validating configuration files...")
        
        config_report = {
            'status': 'pass',
            'files_checked': 0,
            'valid_files': 0,
            'issues': []
        }
        
        config_files = {
            'pyproject.toml': self._validate_pyproject_toml,
            '.pre-commit-config.yaml': self._validate_yaml_file,
            '.github/project-metrics.json': self._validate_json_file,
            'docker-compose.yml': self._validate_yaml_file
        }
        
        for file_path, validator in config_files.items():
            full_path = self.repo_root / file_path
            if full_path.exists():
                config_report['files_checked'] += 1
                try:
                    if validator(full_path):
                        config_report['valid_files'] += 1
                    else:
                        config_report['issues'].append(f'Invalid format: {file_path}')
                        config_report['status'] = 'warning'
                except Exception as e:
                    config_report['issues'].append(f'Error validating {file_path}: {e}')
                    config_report['status'] = 'fail'
        
        return config_report
    
    def validate_testing_setup(self) -> Dict[str, Any]:
        """Validate testing infrastructure."""
        logger.info("Validating testing setup...")
        
        test_report = {
            'status': 'pass',
            'test_files_found': 0,
            'test_categories': {},
            'coverage_configured': False,
            'issues': []
        }
        
        # Check test directories
        test_categories = ['unit', 'integration', 'e2e', 'performance']
        for category in test_categories:
            test_dir = self.repo_root / 'tests' / category
            if test_dir.exists():
                test_files = list(test_dir.glob('test_*.py'))
                test_report['test_categories'][category] = len(test_files)
                test_report['test_files_found'] += len(test_files)
            else:
                test_report['test_categories'][category] = 0
        
        # Check coverage configuration
        coverage_files = ['.coveragerc', 'setup.cfg', 'pyproject.toml']
        for coverage_file in coverage_files:
            if (self.repo_root / coverage_file).exists():
                # Check if it contains coverage configuration
                content = (self.repo_root / coverage_file).read_text()
                if '[coverage:' in content or '[tool.coverage' in content:
                    test_report['coverage_configured'] = True
                    break
        
        if test_report['test_files_found'] == 0:
            test_report['status'] = 'fail'
            test_report['issues'].append('No test files found')
        elif not test_report['coverage_configured']:
            test_report['status'] = 'warning'
            test_report['issues'].append('Code coverage not configured')
        
        return test_report
    
    def validate_ci_cd_setup(self) -> Dict[str, Any]:
        """Validate CI/CD configuration."""
        logger.info("Validating CI/CD setup...")
        
        cicd_report = {
            'status': 'pass',
            'workflows_found': 0,
            'workflow_templates': 0,
            'required_secrets': [],
            'issues': []
        }
        
        # Check GitHub workflows
        workflows_dir = self.repo_root / '.github' / 'workflows'
        if workflows_dir.exists():
            workflow_files = list(workflows_dir.glob('*.yml')) + list(workflows_dir.glob('*.yaml'))
            cicd_report['workflows_found'] = len(workflow_files)
        
        # Check workflow templates
        templates_dir = self.repo_root / 'docs' / 'workflows'
        if templates_dir.exists():
            template_files = list(templates_dir.glob('*.template'))
            cicd_report['workflow_templates'] = len(template_files)
        
        # Check for required secrets documentation
        secrets_indicators = [
            'CODECOV_TOKEN',
            'PYPI_API_TOKEN',
            'SLACK_WEBHOOK_URL',
            'GITHUB_TOKEN'
        ]
        
        # Look for secret references in documentation
        setup_guide = self.repo_root / 'docs' / 'workflows' / 'WORKFLOW_SETUP_GUIDE.md'
        if setup_guide.exists():
            content = setup_guide.read_text()
            for secret in secrets_indicators:
                if secret in content:
                    cicd_report['required_secrets'].append(secret)
        
        if cicd_report['workflows_found'] == 0 and cicd_report['workflow_templates'] == 0:
            cicd_report['status'] = 'fail'
            cicd_report['issues'].append('No CI/CD workflows or templates found')
        elif cicd_report['workflows_found'] == 0:
            cicd_report['status'] = 'warning'
            cicd_report['issues'].append('Workflow templates available but not activated')
        
        return cicd_report
    
    def validate_documentation(self) -> Dict[str, Any]:
        """Validate documentation completeness."""
        logger.info("Validating documentation...")
        
        docs_report = {
            'status': 'pass',
            'essential_docs': {},
            'guide_docs': {},
            'api_docs_configured': False,
            'issues': []
        }
        
        # Check essential documentation
        essential_docs = {
            'README.md': self.repo_root / 'README.md',
            'CONTRIBUTING.md': self.repo_root / 'CONTRIBUTING.md',
            'LICENSE': self.repo_root / 'LICENSE',
            'CHANGELOG.md': self.repo_root / 'CHANGELOG.md'
        }
        
        for doc_name, doc_path in essential_docs.items():
            docs_report['essential_docs'][doc_name] = {
                'exists': doc_path.exists(),
                'size_kb': round(doc_path.stat().st_size / 1024, 1) if doc_path.exists() else 0
            }
        
        # Check guide documentation
        guide_docs = {
            'User Guide': self.repo_root / 'docs' / 'guides' / 'user-guide.md',
            'Developer Guide': self.repo_root / 'docs' / 'guides' / 'developer-guide.md',
            'Deployment Guide': self.repo_root / 'docs' / 'deployment' / 'deployment-guide.md',
            'Monitoring Guide': self.repo_root / 'docs' / 'monitoring' / 'monitoring-guide.md'
        }
        
        for guide_name, guide_path in guide_docs.items():
            docs_report['guide_docs'][guide_name] = guide_path.exists()
        
        # Check for API documentation configuration
        sphinx_conf = self.repo_root / 'docs' / 'conf.py'
        if sphinx_conf.exists():
            docs_report['api_docs_configured'] = True
        
        # Determine status
        missing_essential = sum(1 for doc in docs_report['essential_docs'].values() if not doc['exists'])
        if missing_essential > 2:
            docs_report['status'] = 'fail'
            docs_report['issues'].append(f'{missing_essential} essential documents missing')
        elif missing_essential > 0:
            docs_report['status'] = 'warning'
            docs_report['issues'].append(f'{missing_essential} essential documents missing')
        
        return docs_report
    
    def validate_security_setup(self) -> Dict[str, Any]:
        """Validate security configuration."""
        logger.info("Validating security setup...")
        
        security_report = {
            'status': 'pass',
            'security_files': {},
            'pre_commit_hooks': [],
            'dependency_scanning': False,
            'issues': []
        }
        
        # Check security-related files
        security_files = {
            'SECURITY.md': self.repo_root / 'SECURITY.md',
            '.github/dependabot.yml': self.repo_root / '.github' / 'dependabot.yml',
            '.bandit': self.repo_root / '.bandit'
        }
        
        for file_name, file_path in security_files.items():
            security_report['security_files'][file_name] = file_path.exists()
        
        # Check pre-commit security hooks
        precommit_config = self.repo_root / '.pre-commit-config.yaml'
        if precommit_config.exists():
            try:
                with open(precommit_config, 'r') as f:
                    config = yaml.safe_load(f)
                
                security_hooks = ['bandit', 'safety', 'semgrep']
                for repo in config.get('repos', []):
                    for hook in repo.get('hooks', []):
                        hook_id = hook.get('id', '')
                        for sec_hook in security_hooks:
                            if sec_hook in hook_id:
                                security_report['pre_commit_hooks'].append(hook_id)
            
            except yaml.YAMLError:
                security_report['issues'].append('Invalid pre-commit configuration')
                security_report['status'] = 'warning'
        
        # Check for dependency scanning in workflows
        workflow_templates = self.repo_root / 'docs' / 'workflows'
        if workflow_templates.exists():
            for template_file in workflow_templates.glob('*.template'):
                content = template_file.read_text()
                if 'safety' in content or 'pip-audit' in content or 'snyk' in content:
                    security_report['dependency_scanning'] = True
                    break
        
        # Determine overall status
        if not security_report['security_files']['SECURITY.md']:
            security_report['status'] = 'warning'
            security_report['issues'].append('No security policy documented')
        
        if not security_report['pre_commit_hooks']:
            security_report['status'] = 'warning'
            security_report['issues'].append('No security pre-commit hooks configured')
        
        return security_report
    
    def _validate_pyproject_toml(self, file_path: Path) -> bool:
        """Validate pyproject.toml file."""
        try:
            import tomllib
            with open(file_path, 'rb') as f:
                data = tomllib.load(f)
            
            # Check required sections
            required_sections = ['project', 'build-system']
            for section in required_sections:
                if section not in data:
                    return False
            
            # Check project metadata
            project = data['project']
            required_fields = ['name', 'version', 'description']
            for field in required_fields:
                if field not in project:
                    return False
            
            return True
        except Exception:
            return False
    
    def _validate_yaml_file(self, file_path: Path) -> bool:
        """Validate YAML file syntax."""
        try:
            with open(file_path, 'r') as f:
                yaml.safe_load(f)
            return True
        except yaml.YAMLError:
            return False
    
    def _validate_json_file(self, file_path: Path) -> bool:
        """Validate JSON file syntax."""
        try:
            with open(file_path, 'r') as f:
                json.load(f)
            return True
        except json.JSONDecodeError:
            return False
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation checks."""
        logger.info("Running comprehensive setup validation...")
        
        validation_categories = {
            'project_structure': self.validate_project_structure,
            'python_environment': self.validate_python_environment,
            'configuration_files': self.validate_configuration_files,
            'testing_setup': self.validate_testing_setup,
            'cicd_setup': self.validate_ci_cd_setup,
            'documentation': self.validate_documentation,
            'security_setup': self.validate_security_setup
        }
        
        results = {}
        overall_status = 'pass'
        
        for category, validator in validation_categories.items():
            try:
                result = validator()
                results[category] = result
                
                # Update overall status
                if result['status'] == 'fail':
                    overall_status = 'fail'
                elif result['status'] == 'warning' and overall_status == 'pass':
                    overall_status = 'warning'
                
                # Collect issues
                if 'issues' in result and result['issues']:
                    if result['status'] == 'fail':
                        self.validation_results['critical_issues'].extend(
                            [f"{category}: {issue}" for issue in result['issues']]
                        )
                    else:
                        self.validation_results['warnings'].extend(
                            [f"{category}: {issue}" for issue in result['issues']]
                        )
            
            except Exception as e:
                logger.error(f"Error validating {category}: {e}")
                results[category] = {'status': 'error', 'error': str(e)}
                overall_status = 'fail'
        
        self.validation_results['overall_status'] = overall_status
        self.validation_results['categories'] = results
        
        # Generate recommendations
        self._generate_recommendations()
        
        return self.validation_results
    
    def _generate_recommendations(self):
        """Generate setup recommendations based on validation results."""
        recommendations = []
        
        # Analyze results and generate specific recommendations
        for category, result in self.validation_results['categories'].items():
            if result['status'] in ['fail', 'warning']:
                if category == 'project_structure':
                    if result.get('missing_files'):
                        recommendations.append(
                            f"Create missing files: {', '.join(result['missing_files'])}"
                        )
                
                elif category == 'python_environment':
                    if not result.get('dependencies_installed'):
                        recommendations.append("Install project dependencies: pip install -e .[dev]")
                
                elif category == 'testing_setup':
                    if result.get('test_files_found', 0) == 0:
                        recommendations.append("Create test files in tests/ directory")
                    if not result.get('coverage_configured'):
                        recommendations.append("Configure code coverage in pyproject.toml")
                
                elif category == 'cicd_setup':
                    if result.get('workflows_found', 0) == 0:
                        recommendations.append("Set up GitHub Actions workflows from templates")
                
                elif category == 'documentation':
                    missing_docs = [doc for doc, info in result.get('essential_docs', {}).items() 
                                  if not info.get('exists')]
                    if missing_docs:
                        recommendations.append(f"Create missing documentation: {', '.join(missing_docs)}")
                
                elif category == 'security_setup':
                    if not result.get('security_files', {}).get('SECURITY.md'):
                        recommendations.append("Create SECURITY.md file")
                    if not result.get('pre_commit_hooks'):
                        recommendations.append("Add security hooks to pre-commit configuration")
        
        self.validation_results['recommendations'] = recommendations


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Validate project setup')
    parser.add_argument('--category', 
                       choices=['structure', 'python', 'config', 'testing', 'cicd', 'docs', 'security', 'all'],
                       default='all',
                       help='Validation category to run')
    parser.add_argument('--output', help='Output file for validation report')
    parser.add_argument('--format', choices=['json', 'summary'], default='summary',
                       help='Output format')
    
    args = parser.parse_args()
    
    try:
        validator = SetupValidator()
        
        if args.category == 'all':
            result = validator.run_comprehensive_validation()
        else:
            # Run specific category
            category_map = {
                'structure': validator.validate_project_structure,
                'python': validator.validate_python_environment,
                'config': validator.validate_configuration_files,
                'testing': validator.validate_testing_setup,
                'cicd': validator.validate_ci_cd_setup,
                'docs': validator.validate_documentation,
                'security': validator.validate_security_setup
            }
            result = category_map[args.category]()
        
        if args.format == 'json':
            output = json.dumps(result, indent=2)
        else:
            # Generate summary format
            if args.category == 'all':
                output = _generate_summary_report(result)
            else:
                output = json.dumps(result, indent=2)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            logger.info(f"Validation report saved to {args.output}")
        else:
            print(output)
        
        # Exit with appropriate code
        if result.get('overall_status') == 'fail' or (isinstance(result, dict) and result.get('status') == 'fail'):
            sys.exit(1)
        elif result.get('overall_status') == 'warning' or (isinstance(result, dict) and result.get('status') == 'warning'):
            sys.exit(2)
        else:
            sys.exit(0)
    
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)


def _generate_summary_report(results: Dict[str, Any]) -> str:
    """Generate human-readable summary report."""
    lines = []
    lines.append("# Project Setup Validation Report")
    lines.append("")
    lines.append(f"**Overall Status:** {results['overall_status'].upper()}")
    lines.append("")
    
    # Category results
    lines.append("## Validation Results")
    lines.append("")
    for category, result in results['categories'].items():
        status_icon = {"pass": "✅", "warning": "⚠️", "fail": "❌", "error": "💥"}
        icon = status_icon.get(result['status'], "❓")
        lines.append(f"- **{category.replace('_', ' ').title()}**: {icon} {result['status'].upper()}")
    
    lines.append("")
    
    # Critical issues
    if results['critical_issues']:
        lines.append("## Critical Issues")
        lines.append("")
        for issue in results['critical_issues']:
            lines.append(f"- ❌ {issue}")
        lines.append("")
    
    # Warnings
    if results['warnings']:
        lines.append("## Warnings")
        lines.append("")
        for warning in results['warnings']:
            lines.append(f"- ⚠️ {warning}")
        lines.append("")
    
    # Recommendations
    if results['recommendations']:
        lines.append("## Recommendations")
        lines.append("")
        for i, rec in enumerate(results['recommendations'], 1):
            lines.append(f"{i}. {rec}")
        lines.append("")
    
    lines.append("---")
    lines.append("*Generated by HF Eco2AI Plugin setup validator*")
    
    return "\n".join(lines)


if __name__ == '__main__':
    main()