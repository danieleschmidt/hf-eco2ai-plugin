#!/usr/bin/env python3
"""Automated metrics collection script for HF Eco2AI Plugin."""

import json
import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
import requests
import subprocess
import tempfile

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collect and update project metrics."""
    
    def __init__(self, config_path: str = ".github/project-metrics.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.repo_owner = self.config['repository']['owner']
        self.repo_name = self.config['repository']['name']
    
    def _load_config(self) -> Dict[str, Any]:
        """Load metrics configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Metrics config not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def _save_config(self) -> None:
        """Save updated metrics configuration."""
        self.config['lastUpdated'] = datetime.utcnow().isoformat() + 'Z'
        
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Updated metrics saved to {self.config_path}")
    
    def _github_api_request(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """Make GitHub API request."""
        if not self.github_token:
            logger.warning("GitHub token not available, skipping GitHub metrics")
            return None
        
        headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        try:
            response = requests.get(endpoint, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"GitHub API request failed: {e}")
            return None
    
    def collect_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics."""
        logger.info("Collecting code quality metrics...")
        metrics = {}
        
        try:
            # Test coverage from coverage.xml
            coverage_file = Path('coverage.xml')
            if coverage_file.exists():
                import xml.etree.ElementTree as ET
                tree = ET.parse(coverage_file)
                root = tree.getroot()
                line_rate = float(root.attrib.get('line-rate', 0))
                metrics['test_coverage'] = {
                    'value': round(line_rate * 100, 1),
                    'unit': 'percent',
                    'trend': 'stable',  # Would need historical data to determine
                    'last_measurement': datetime.utcnow().isoformat() + 'Z'
                }
            
            # Code complexity using radon
            try:
                result = subprocess.run(
                    ['radon', 'cc', 'src/', '--average', '--json'],
                    capture_output=True, text=True, timeout=60
                )
                if result.returncode == 0:
                    radon_data = json.loads(result.stdout)
                    avg_complexity = sum(
                        file_data.get('average_complexity', 0) 
                        for file_data in radon_data.values()
                    ) / len(radon_data) if radon_data else 0
                    
                    metrics['code_complexity'] = {
                        'value': round(avg_complexity, 1),
                        'unit': 'cyclomatic_complexity',
                        'trend': 'stable',
                        'last_measurement': datetime.utcnow().isoformat() + 'Z'
                    }
            except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
                logger.warning("Could not collect code complexity metrics")
            
            # Lines of code
            try:
                result = subprocess.run(
                    ['cloc', 'src/', '--json'],
                    capture_output=True, text=True, timeout=60
                )
                if result.returncode == 0:
                    cloc_data = json.loads(result.stdout)
                    total_lines = cloc_data.get('SUM', {}).get('code', 0)
                    metrics['lines_of_code'] = {
                        'value': total_lines,
                        'unit': 'lines',
                        'trend': 'increasing',
                        'last_measurement': datetime.utcnow().isoformat() + 'Z'
                    }
            except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
                logger.warning("Could not collect lines of code metrics")
        
        except Exception as e:
            logger.error(f"Error collecting code quality metrics: {e}")
        
        return metrics
    
    def collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security metrics."""
        logger.info("Collecting security metrics...")
        metrics = {}
        
        try:
            # Security vulnerabilities from safety
            try:
                result = subprocess.run(
                    ['safety', 'check', '--json'],
                    capture_output=True, text=True, timeout=120
                )
                if result.returncode == 0:
                    safety_data = json.loads(result.stdout)
                    critical_count = sum(
                        1 for vuln in safety_data 
                        if vuln.get('severity', '').lower() == 'critical'
                    )
                    high_count = sum(
                        1 for vuln in safety_data 
                        if vuln.get('severity', '').lower() == 'high'
                    )
                    
                    metrics.update({
                        'critical_vulnerabilities': {
                            'value': critical_count,
                            'unit': 'count',
                            'trend': 'stable',
                            'last_measurement': datetime.utcnow().isoformat() + 'Z'
                        },
                        'high_vulnerabilities': {
                            'value': high_count,
                            'unit': 'count',
                            'trend': 'stable',
                            'last_measurement': datetime.utcnow().isoformat() + 'Z'
                        }
                    })
            except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
                logger.warning("Could not collect safety metrics")
            
            # Last security scan timestamp
            metrics['last_security_scan'] = {
                'value': datetime.utcnow().isoformat() + 'Z',
                'unit': 'timestamp',
                'description': 'Last comprehensive security scan'
            }
        
        except Exception as e:
            logger.error(f"Error collecting security metrics: {e}")
        
        return metrics
    
    def collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics."""
        logger.info("Collecting performance metrics...")
        metrics = {}
        
        try:
            # Check if benchmark results exist
            benchmark_file = Path('benchmark-results.json')
            if benchmark_file.exists():
                with open(benchmark_file, 'r') as f:
                    benchmark_data = json.load(f)
                
                # Extract callback overhead from benchmarks
                for benchmark in benchmark_data.get('benchmarks', []):
                    if 'callback_overhead' in benchmark.get('name', ''):
                        metrics['callback_overhead'] = {
                            'value': round(benchmark.get('stats', {}).get('mean', 0) * 100, 1),
                            'unit': 'percent',
                            'trend': 'stable',
                            'last_measurement': datetime.utcnow().isoformat() + 'Z'
                        }
                        break
            
            # Memory usage from process monitoring
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                metrics['memory_usage'] = {
                    'value': round(memory_mb, 1),
                    'unit': 'megabytes',
                    'trend': 'stable',
                    'last_measurement': datetime.utcnow().isoformat() + 'Z'
                }
            except ImportError:
                logger.warning("psutil not available for memory metrics")
        
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
        
        return metrics
    
    def collect_github_metrics(self) -> Dict[str, Any]:
        """Collect GitHub repository metrics."""
        logger.info("Collecting GitHub metrics...")
        metrics = {}
        
        try:
            # Repository basic info
            repo_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}"
            repo_data = self._github_api_request(repo_url)
            
            if repo_data:
                metrics.update({
                    'github_stars': {
                        'value': repo_data.get('stargazers_count', 0),
                        'unit': 'count',
                        'trend': 'increasing',
                        'last_measurement': datetime.utcnow().isoformat() + 'Z'
                    },
                    'github_forks': {
                        'value': repo_data.get('forks_count', 0),
                        'unit': 'count',
                        'trend': 'increasing',
                        'last_measurement': datetime.utcnow().isoformat() + 'Z'
                    },
                    'open_issues': {
                        'value': repo_data.get('open_issues_count', 0),
                        'unit': 'count',
                        'trend': 'stable',
                        'last_measurement': datetime.utcnow().isoformat() + 'Z'
                    }
                })
            
            # Pull requests
            prs_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/pulls"
            prs_data = self._github_api_request(prs_url)
            
            if prs_data:
                metrics['open_pull_requests'] = {
                    'value': len(prs_data),
                    'unit': 'count',
                    'trend': 'stable',
                    'last_measurement': datetime.utcnow().isoformat() + 'Z'
                }
            
            # Build success rate from recent workflow runs
            runs_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/actions/runs"
            runs_data = self._github_api_request(runs_url)
            
            if runs_data and runs_data.get('workflow_runs'):
                recent_runs = runs_data['workflow_runs'][:50]  # Last 50 runs
                successful_runs = sum(
                    1 for run in recent_runs 
                    if run.get('conclusion') == 'success'
                )
                success_rate = (successful_runs / len(recent_runs)) * 100 if recent_runs else 0
                
                metrics['build_success_rate'] = {
                    'value': round(success_rate, 1),
                    'unit': 'percent',
                    'trend': 'stable',
                    'last_measurement': datetime.utcnow().isoformat() + 'Z'
                }
                
                # Average build time
                durations = []
                for run in recent_runs:
                    if run.get('created_at') and run.get('updated_at'):
                        start = datetime.fromisoformat(run['created_at'].replace('Z', '+00:00'))
                        end = datetime.fromisoformat(run['updated_at'].replace('Z', '+00:00'))
                        duration = (end - start).total_seconds() / 60  # minutes
                        durations.append(duration)
                
                if durations:
                    avg_duration = sum(durations) / len(durations)
                    metrics['average_build_time'] = {
                        'value': round(avg_duration, 1),
                        'unit': 'minutes',
                        'trend': 'improving',
                        'last_measurement': datetime.utcnow().isoformat() + 'Z'
                    }
        
        except Exception as e:
            logger.error(f"Error collecting GitHub metrics: {e}")
        
        return metrics
    
    def collect_pypi_metrics(self) -> Dict[str, Any]:
        """Collect PyPI download metrics."""
        logger.info("Collecting PyPI metrics...")
        metrics = {}
        
        try:
            # PyPI download statistics
            package_name = 'hf-eco2ai-plugin'
            stats_url = f"https://pypistats.org/api/packages/{package_name}/recent"
            
            response = requests.get(stats_url, timeout=30)
            if response.status_code == 200:
                stats_data = response.json()
                last_month = stats_data.get('data', {}).get('last_month', 0)
                
                metrics['pypi_downloads_monthly'] = {
                    'value': last_month,
                    'unit': 'count',
                    'trend': 'increasing',
                    'last_measurement': datetime.utcnow().isoformat() + 'Z'
                }
        
        except Exception as e:
            logger.error(f"Error collecting PyPI metrics: {e}")
        
        return metrics
    
    def collect_sustainability_metrics(self) -> Dict[str, Any]:
        """Collect sustainability metrics."""
        logger.info("Collecting sustainability metrics...")
        metrics = {}
        
        try:
            # Check for carbon tracking data
            carbon_reports = list(Path('.').glob('**/carbon_report.json'))
            if carbon_reports:
                with open(carbon_reports[0], 'r') as f:
                    carbon_data = json.load(f)
                
                metrics.update({
                    'total_energy_tracked': {
                        'value': carbon_data.get('total_energy_kwh', 0),
                        'unit': 'kwh',
                        'trend': 'increasing',
                        'last_measurement': datetime.utcnow().isoformat() + 'Z'
                    },
                    'monthly_carbon_usage': {
                        'value': carbon_data.get('total_co2_kg', 0),
                        'unit': 'kg_co2',
                        'trend': 'stable',
                        'last_measurement': datetime.utcnow().isoformat() + 'Z'
                    },
                    'current_carbon_intensity': {
                        'value': carbon_data.get('grid_carbon_intensity', 400),
                        'unit': 'g_co2_per_kwh',
                        'trend': 'variable',
                        'last_measurement': datetime.utcnow().isoformat() + 'Z'
                    }
                })
                
                # Calculate efficiency score
                if carbon_data.get('samples_per_kwh'):
                    efficiency_score = min(carbon_data['samples_per_kwh'] / 5000, 1.0)
                    metrics['carbon_efficiency_score'] = {
                        'value': round(efficiency_score, 2),
                        'unit': 'score',
                        'trend': 'improving',
                        'last_measurement': datetime.utcnow().isoformat() + 'Z'
                    }
        
        except Exception as e:
            logger.error(f"Error collecting sustainability metrics: {e}")
        
        return metrics
    
    def update_metrics(self) -> None:
        """Update all metrics in the configuration."""
        logger.info("Starting metrics collection...")
        
        # Collect all metric categories
        metric_collectors = {
            'code_quality': self.collect_code_quality_metrics,
            'security': self.collect_security_metrics,
            'performance': self.collect_performance_metrics,
            'adoption': lambda: {**self.collect_github_metrics(), **self.collect_pypi_metrics()},
            'development': self.collect_github_metrics,
            'sustainability': self.collect_sustainability_metrics
        }
        
        for category, collector in metric_collectors.items():
            try:
                logger.info(f"Collecting {category} metrics...")
                new_metrics = collector()
                
                # Update current metrics in config
                if category in self.config['metrics']:
                    self.config['metrics'][category]['current'].update(new_metrics)
                
                logger.info(f"Updated {len(new_metrics)} {category} metrics")
                
            except Exception as e:
                logger.error(f"Failed to collect {category} metrics: {e}")
        
        # Save updated configuration
        self._save_config()
        logger.info("Metrics collection completed")
    
    def generate_report(self, format: str = 'json') -> str:
        """Generate metrics report."""
        if format == 'json':
            return json.dumps(self.config, indent=2)
        elif format == 'summary':
            return self._generate_summary_report()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _generate_summary_report(self) -> str:
        """Generate human-readable summary report."""
        report = []
        report.append(f"# HF Eco2AI Plugin Metrics Summary")
        report.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report.append("")
        
        for category, data in self.config['metrics'].items():
            report.append(f"## {category.replace('_', ' ').title()}")
            report.append("")
            
            current = data.get('current', {})
            targets = data.get('targets', {})
            
            for metric, values in current.items():
                value = values.get('value', 'N/A')
                unit = values.get('unit', '')
                trend = values.get('trend', 'unknown')
                target_value = targets.get(metric, {}).get('value', 'N/A')
                
                status = "✅" if self._is_metric_on_target(metric, values, targets.get(metric, {})) else "⚠️"
                
                report.append(f"- **{metric.replace('_', ' ').title()}**: {value} {unit} (Target: {target_value}) {status}")
                report.append(f"  Trend: {trend}")
            
            report.append("")
        
        return "\n".join(report)
    
    def _is_metric_on_target(self, metric_name: str, current: Dict, target: Dict) -> bool:
        """Check if metric is meeting target."""
        if not target or 'value' not in current or 'value' not in target:
            return True  # No target defined or missing data
        
        current_value = current['value']
        target_value = target['value']
        
        # Define metrics where higher is better
        higher_is_better = {
            'test_coverage', 'build_success_rate', 'carbon_efficiency_score',
            'renewable_energy_ratio', 'github_stars', 'pypi_downloads_monthly'
        }
        
        if metric_name in higher_is_better:
            return current_value >= target_value
        else:
            return current_value <= target_value


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect project metrics')
    parser.add_argument('--config', default='.github/project-metrics.json',
                       help='Path to metrics configuration file')
    parser.add_argument('--format', choices=['json', 'summary'], default='summary',
                       help='Output format')
    parser.add_argument('--output', help='Output file (default: stdout)')
    parser.add_argument('--update', action='store_true',
                       help='Update metrics in configuration file')
    
    args = parser.parse_args()
    
    try:
        collector = MetricsCollector(args.config)
        
        if args.update:
            collector.update_metrics()
        
        report = collector.generate_report(args.format)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {args.output}")
        else:
            print(report)
    
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
