#!/usr/bin/env python3
"""Repository maintenance automation script for HF Eco2AI Plugin."""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import tempfile

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RepositoryMaintenance:
    """Automate repository maintenance tasks."""
    
    def __init__(self):
        self.repo_root = Path.cwd()
        self.github_token = os.getenv('GITHUB_TOKEN')
        
    def cleanup_branches(self, dry_run: bool = True) -> Dict[str, Any]:
        """Clean up merged and stale branches."""
        logger.info("Cleaning up branches...")
        
        try:
            # Get merged branches
            result = subprocess.run(
                ['git', 'branch', '--merged', 'main'],
                capture_output=True, text=True, check=True
            )
            
            merged_branches = [
                line.strip().replace('*', '').strip()
                for line in result.stdout.splitlines()
                if line.strip() and not line.strip().startswith('main')
            ]
            
            # Get remote tracking info
            result = subprocess.run(
                ['git', 'branch', '-vv'],
                capture_output=True, text=True, check=True
            )
            
            stale_branches = []
            for line in result.stdout.splitlines():
                if ': gone]' in line:
                    branch = line.split()[0].replace('*', '').strip()
                    if branch != 'main':
                        stale_branches.append(branch)
            
            cleanup_report = {
                'merged_branches': merged_branches,
                'stale_branches': stale_branches,
                'total_cleanup': len(merged_branches) + len(stale_branches),
                'dry_run': dry_run
            }
            
            if not dry_run:
                # Delete merged branches
                for branch in merged_branches:
                    try:
                        subprocess.run(['git', 'branch', '-d', branch], check=True)
                        logger.info(f"Deleted merged branch: {branch}")
                    except subprocess.CalledProcessError as e:
                        logger.warning(f"Failed to delete branch {branch}: {e}")
                
                # Delete stale branches
                for branch in stale_branches:
                    try:
                        subprocess.run(['git', 'branch', '-D', branch], check=True)
                        logger.info(f"Deleted stale branch: {branch}")
                    except subprocess.CalledProcessError as e:
                        logger.warning(f"Failed to delete branch {branch}: {e}")
            
            return cleanup_report
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error during branch cleanup: {e}")
            return {'error': str(e)}
    
    def update_dependencies(self, update_type: str = 'patch') -> Dict[str, Any]:
        """Update project dependencies."""
        logger.info(f"Updating {update_type} dependencies...")
        
        try:
            # Check for outdated packages
            result = subprocess.run(
                ['pip', 'list', '--outdated', '--format=json'],
                capture_output=True, text=True, check=True
            )
            
            outdated = json.loads(result.stdout)
            
            update_report = {
                'update_type': update_type,
                'outdated_count': len(outdated),
                'updated_packages': [],
                'failed_packages': []
            }
            
            for package in outdated:
                name = package['name']
                current = package['version']
                latest = package['latest_version']
                
                # Simple version comparison for update type
                current_parts = current.split('.')
                latest_parts = latest.split('.')
                
                should_update = False
                if update_type == 'patch' and len(current_parts) >= 3 and len(latest_parts) >= 3:
                    should_update = (current_parts[0] == latest_parts[0] and 
                                   current_parts[1] == latest_parts[1])
                elif update_type == 'minor' and len(current_parts) >= 2 and len(latest_parts) >= 2:
                    should_update = current_parts[0] == latest_parts[0]
                elif update_type == 'all':
                    should_update = True
                
                if should_update:
                    try:
                        subprocess.run(
                            ['pip', 'install', '--upgrade', f'{name}=={latest}'],
                            check=True, capture_output=True
                        )
                        update_report['updated_packages'].append({
                            'name': name,
                            'from': current,
                            'to': latest
                        })
                        logger.info(f"Updated {name}: {current} -> {latest}")
                    except subprocess.CalledProcessError:
                        update_report['failed_packages'].append(name)
                        logger.warning(f"Failed to update {name}")
            
            return update_report
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error updating dependencies: {e}")
            return {'error': str(e)}
    
    def cleanup_artifacts(self, max_age_days: int = 30) -> Dict[str, Any]:
        """Clean up old build artifacts and temporary files."""
        logger.info(f"Cleaning up artifacts older than {max_age_days} days...")
        
        cleanup_patterns = [
            '**/__pycache__',
            '**/*.pyc',
            '**/*.pyo',
            '**/.*cache*',
            'dist/*',
            'build/*',
            '**/.pytest_cache',
            '**/htmlcov',
            '**/*.log',
            'coverage.xml',
            '.coverage*'
        ]
        
        cleanup_report = {
            'cleaned_files': [],
            'cleaned_dirs': [],
            'total_size_mb': 0,
            'max_age_days': max_age_days
        }
        
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        try:
            for pattern in cleanup_patterns:
                for path in self.repo_root.glob(pattern):
                    if path.exists():
                        try:
                            # Check file age
                            if path.stat().st_mtime < cutoff_date.timestamp():
                                if path.is_file():
                                    size_mb = path.stat().st_size / (1024 * 1024)
                                    path.unlink()
                                    cleanup_report['cleaned_files'].append(str(path))
                                    cleanup_report['total_size_mb'] += size_mb
                                elif path.is_dir():
                                    import shutil
                                    shutil.rmtree(path)
                                    cleanup_report['cleaned_dirs'].append(str(path))
                        except (OSError, PermissionError) as e:
                            logger.warning(f"Could not clean {path}: {e}")
            
            cleanup_report['total_size_mb'] = round(cleanup_report['total_size_mb'], 2)
            return cleanup_report
            
        except Exception as e:
            logger.error(f"Error during artifact cleanup: {e}")
            return {'error': str(e)}
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate repository health report."""
        logger.info("Generating repository health report...")
        
        health_report = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'repository': {
                'name': 'hf-eco2ai-plugin',
                'status': 'healthy'
            },
            'metrics': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Check git status
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True, text=True, check=True
            )
            
            uncommitted_files = len(result.stdout.splitlines())
            health_report['metrics']['uncommitted_files'] = uncommitted_files
            
            if uncommitted_files > 0:
                health_report['issues'].append(f"{uncommitted_files} uncommitted files")
            
            # Check for large files
            large_files = []
            for file_path in self.repo_root.rglob('*'):
                if file_path.is_file() and file_path.stat().st_size > 10 * 1024 * 1024:  # 10MB
                    large_files.append(str(file_path))
            
            health_report['metrics']['large_files'] = len(large_files)
            if large_files:
                health_report['issues'].append(f"{len(large_files)} large files detected")
                health_report['recommendations'].append("Consider using Git LFS for large files")
            
            # Check test coverage
            coverage_file = self.repo_root / 'coverage.xml'
            if coverage_file.exists():
                try:
                    import xml.etree.ElementTree as ET
                    tree = ET.parse(coverage_file)
                    root = tree.getroot()
                    line_rate = float(root.attrib.get('line-rate', 0))
                    coverage_percent = round(line_rate * 100, 1)
                    health_report['metrics']['test_coverage'] = coverage_percent
                    
                    if coverage_percent < 80:
                        health_report['issues'].append(f"Low test coverage: {coverage_percent}%")
                        health_report['recommendations'].append("Increase test coverage to at least 80%")
                except Exception:
                    health_report['metrics']['test_coverage'] = 'unknown'
            
            # Overall health status
            if len(health_report['issues']) == 0:
                health_report['repository']['status'] = 'excellent'
            elif len(health_report['issues']) <= 2:
                health_report['repository']['status'] = 'good'
            elif len(health_report['issues']) <= 5:
                health_report['repository']['status'] = 'fair'
            else:
                health_report['repository']['status'] = 'needs_attention'
            
            return health_report
            
        except Exception as e:
            logger.error(f"Error generating health report: {e}")
            health_report['error'] = str(e)
            health_report['repository']['status'] = 'unknown'
            return health_report
    
    def run_full_maintenance(self, dry_run: bool = True) -> Dict[str, Any]:
        """Run complete maintenance suite."""
        logger.info("Running full maintenance suite...")
        
        maintenance_report = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'dry_run': dry_run,
            'tasks': {}
        }
        
        # Run all maintenance tasks
        maintenance_report['tasks']['branch_cleanup'] = self.cleanup_branches(dry_run)
        maintenance_report['tasks']['dependency_updates'] = self.update_dependencies('patch')
        maintenance_report['tasks']['artifact_cleanup'] = self.cleanup_artifacts()
        maintenance_report['tasks']['health_report'] = self.generate_health_report()
        
        # Summary
        total_issues = len(maintenance_report['tasks']['health_report'].get('issues', []))
        maintenance_report['summary'] = {
            'status': 'completed',
            'total_issues': total_issues,
            'recommendation_count': len(maintenance_report['tasks']['health_report'].get('recommendations', []))
        }
        
        return maintenance_report


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Repository maintenance automation')
    parser.add_argument('--task', choices=['branches', 'deps', 'artifacts', 'health', 'full'],
                       default='full', help='Maintenance task to run')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without making changes')
    parser.add_argument('--update-type', choices=['patch', 'minor', 'all'],
                       default='patch', help='Type of dependency updates')
    parser.add_argument('--output', help='Output file for reports')
    
    args = parser.parse_args()
    
    try:
        maintenance = RepositoryMaintenance()
        
        if args.task == 'branches':
            result = maintenance.cleanup_branches(args.dry_run)
        elif args.task == 'deps':
            result = maintenance.update_dependencies(args.update_type)
        elif args.task == 'artifacts':
            result = maintenance.cleanup_artifacts()
        elif args.task == 'health':
            result = maintenance.generate_health_report()
        else:  # full
            result = maintenance.run_full_maintenance(args.dry_run)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Report saved to {args.output}")
        else:
            print(json.dumps(result, indent=2))
    
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()