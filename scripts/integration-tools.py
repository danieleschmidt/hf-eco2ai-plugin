#!/usr/bin/env python3
"""Integration scripts for external tools and services."""

import argparse
import json
import logging
import os
import requests
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExternalIntegrations:
    """Manage integrations with external tools and services."""
    
    def __init__(self):
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.slack_webhook = os.getenv('SLACK_WEBHOOK_URL')
        self.codecov_token = os.getenv('CODECOV_TOKEN')
        
    def sync_github_labels(self) -> Dict[str, Any]:
        """Synchronize GitHub repository labels."""
        logger.info("Synchronizing GitHub labels...")
        
        if not self.github_token:
            return {'error': 'GitHub token not available'}
        
        # Standard labels configuration
        standard_labels = [
            {'name': 'bug', 'color': 'd73a4a', 'description': 'Something isn\'t working'},
            {'name': 'enhancement', 'color': 'a2eeef', 'description': 'New feature or request'},
            {'name': 'documentation', 'color': '0075ca', 'description': 'Improvements or additions to documentation'},
            {'name': 'good first issue', 'color': '7057ff', 'description': 'Good for newcomers'},
            {'name': 'help wanted', 'color': '008672', 'description': 'Extra attention is needed'},
            {'name': 'dependencies', 'color': '0366d6', 'description': 'Pull requests that update a dependency file'},
            {'name': 'security', 'color': 'ee0701', 'description': 'Security-related issues'},
            {'name': 'performance', 'color': 'ffcc00', 'description': 'Performance improvements'},
            {'name': 'testing', 'color': '1a8f47', 'description': 'Related to testing'},
            {'name': 'ci/cd', 'color': '34495e', 'description': 'Continuous integration and deployment'},
            {'name': 'carbon-tracking', 'color': '2ecc71', 'description': 'Carbon footprint and sustainability'},
            {'name': 'sustainability', 'color': '27ae60', 'description': 'Environmental sustainability'},
            {'name': 'automated', 'color': '95a5a6', 'description': 'Automated by bots or workflows'},
            {'name': 'priority: critical', 'color': 'b60205', 'description': 'Critical priority'},
            {'name': 'priority: high', 'color': 'ff9800', 'description': 'High priority'},
            {'name': 'priority: medium', 'color': 'ffeb3b', 'description': 'Medium priority'},
            {'name': 'priority: low', 'color': '4caf50', 'description': 'Low priority'},
        ]
        
        headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        repo_url = 'https://api.github.com/repos/danieleschmidt/hf-eco2ai-plugin'
        
        sync_report = {
            'created': [],
            'updated': [],
            'errors': []
        }
        
        try:
            # Get existing labels
            response = requests.get(f'{repo_url}/labels', headers=headers)
            response.raise_for_status()
            existing_labels = {label['name']: label for label in response.json()}
            
            for label_config in standard_labels:
                label_name = label_config['name']
                
                if label_name in existing_labels:
                    # Update existing label
                    existing = existing_labels[label_name]
                    if (existing['color'] != label_config['color'] or 
                        existing.get('description', '') != label_config['description']):
                        
                        update_response = requests.patch(
                            f'{repo_url}/labels/{label_name}',
                            headers=headers,
                            json=label_config
                        )
                        
                        if update_response.status_code == 200:
                            sync_report['updated'].append(label_name)
                            logger.info(f"Updated label: {label_name}")
                        else:
                            sync_report['errors'].append(f"Failed to update {label_name}")
                else:
                    # Create new label
                    create_response = requests.post(
                        f'{repo_url}/labels',
                        headers=headers,
                        json=label_config
                    )
                    
                    if create_response.status_code == 201:
                        sync_report['created'].append(label_name)
                        logger.info(f"Created label: {label_name}")
                    else:
                        sync_report['errors'].append(f"Failed to create {label_name}")
            
            return sync_report
            
        except requests.RequestException as e:
            logger.error(f"Error syncing GitHub labels: {e}")
            return {'error': str(e)}
    
    def upload_coverage_report(self, coverage_file: str = 'coverage.xml') -> Dict[str, Any]:
        """Upload coverage report to Codecov."""
        logger.info("Uploading coverage report to Codecov...")
        
        if not self.codecov_token:
            return {'error': 'Codecov token not available'}
        
        coverage_path = Path(coverage_file)
        if not coverage_path.exists():
            return {'error': f'Coverage file not found: {coverage_file}'}
        
        try:
            # Use codecov uploader
            env = os.environ.copy()
            env['CODECOV_TOKEN'] = self.codecov_token
            
            result = subprocess.run(
                ['bash', '-c', 'curl -s https://codecov.io/bash | bash'],
                env=env,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            upload_report = {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            if result.returncode == 0:
                logger.info("Coverage report uploaded successfully")
            else:
                logger.error(f"Coverage upload failed: {result.stderr}")
            
            return upload_report
            
        except subprocess.TimeoutExpired:
            return {'error': 'Coverage upload timed out'}
        except Exception as e:
            logger.error(f"Error uploading coverage: {e}")
            return {'error': str(e)}
    
    def send_slack_notification(self, message: str, channel: str = None) -> Dict[str, Any]:
        """Send notification to Slack."""
        logger.info("Sending Slack notification...")
        
        if not self.slack_webhook:
            return {'error': 'Slack webhook not configured'}
        
        payload = {
            'text': message,
            'username': 'HF Eco2AI Bot',
            'icon_emoji': ':robot_face:'
        }
        
        if channel:
            payload['channel'] = channel
        
        try:
            response = requests.post(
                self.slack_webhook,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            return {
                'success': True,
                'status_code': response.status_code,
                'message': 'Notification sent successfully'
            }
            
        except requests.RequestException as e:
            logger.error(f"Error sending Slack notification: {e}")
            return {'error': str(e)}
    
    def sync_project_boards(self) -> Dict[str, Any]:
        """Synchronize GitHub project boards."""
        logger.info("Synchronizing GitHub project boards...")
        
        if not self.github_token:
            return {'error': 'GitHub token not available'}
        
        headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.inertia-preview+json'
        }
        
        repo_url = 'https://api.github.com/repos/danieleschmidt/hf-eco2ai-plugin'
        
        # Project board configuration
        project_config = {
            'name': 'HF Eco2AI Development',
            'body': 'Main development board for HF Eco2AI Plugin',
            'columns': [
                {'name': 'Backlog', 'preset': None},
                {'name': 'To Do', 'preset': None},
                {'name': 'In Progress', 'preset': None},
                {'name': 'Review', 'preset': None},
                {'name': 'Done', 'preset': None}
            ]
        }
        
        try:
            # Check existing projects
            response = requests.get(f'{repo_url}/projects', headers=headers)
            response.raise_for_status()
            existing_projects = response.json()
            
            target_project = None
            for project in existing_projects:
                if project['name'] == project_config['name']:
                    target_project = project
                    break
            
            sync_report = {'project_id': None, 'columns': []}
            
            if not target_project:
                # Create project
                create_response = requests.post(
                    f'{repo_url}/projects',
                    headers=headers,
                    json={
                        'name': project_config['name'],
                        'body': project_config['body']
                    }
                )
                
                if create_response.status_code == 201:
                    target_project = create_response.json()
                    logger.info(f"Created project board: {project_config['name']}")
                else:
                    return {'error': 'Failed to create project board'}
            
            sync_report['project_id'] = target_project['id']
            
            # Get existing columns
            columns_response = requests.get(
                f"https://api.github.com/projects/{target_project['id']}/columns",
                headers=headers
            )
            columns_response.raise_for_status()
            existing_columns = {col['name']: col for col in columns_response.json()}
            
            # Create missing columns
            for col_config in project_config['columns']:
                if col_config['name'] not in existing_columns:
                    create_col_response = requests.post(
                        f"https://api.github.com/projects/{target_project['id']}/columns",
                        headers=headers,
                        json={'name': col_config['name']}
                    )
                    
                    if create_col_response.status_code == 201:
                        sync_report['columns'].append(col_config['name'])
                        logger.info(f"Created column: {col_config['name']}")
            
            return sync_report
            
        except requests.RequestException as e:
            logger.error(f"Error syncing project boards: {e}")
            return {'error': str(e)}
    
    def generate_integration_report(self) -> Dict[str, Any]:
        """Generate comprehensive integration status report."""
        logger.info("Generating integration status report...")
        
        report = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'integrations': {},
            'summary': {}
        }
        
        # Test each integration
        integrations = [
            ('github_labels', self.sync_github_labels),
            ('slack_notification', lambda: self.send_slack_notification('Integration test')),
            ('project_boards', self.sync_project_boards)
        ]
        
        successful = 0
        total = len(integrations)
        
        for name, test_func in integrations:
            try:
                result = test_func()
                report['integrations'][name] = {
                    'status': 'success' if 'error' not in result else 'error',
                    'details': result
                }
                
                if 'error' not in result:
                    successful += 1
                    
            except Exception as e:
                report['integrations'][name] = {
                    'status': 'error',
                    'details': {'error': str(e)}
                }
        
        report['summary'] = {
            'total_integrations': total,
            'successful': successful,
            'failed': total - successful,
            'success_rate': round((successful / total) * 100, 1) if total > 0 else 0
        }
        
        return report


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='External integrations management')
    parser.add_argument('--action', 
                       choices=['labels', 'coverage', 'slack', 'boards', 'report', 'all'],
                       default='report',
                       help='Integration action to perform')
    parser.add_argument('--message', help='Message for Slack notification')
    parser.add_argument('--coverage-file', default='coverage.xml',
                       help='Coverage file path')
    parser.add_argument('--output', help='Output file for reports')
    
    args = parser.parse_args()
    
    try:
        integrations = ExternalIntegrations()
        
        if args.action == 'labels':
            result = integrations.sync_github_labels()
        elif args.action == 'coverage':
            result = integrations.upload_coverage_report(args.coverage_file)
        elif args.action == 'slack':
            if not args.message:
                logger.error("Message required for Slack notification")
                sys.exit(1)
            result = integrations.send_slack_notification(args.message)
        elif args.action == 'boards':
            result = integrations.sync_project_boards()
        elif args.action == 'report':
            result = integrations.generate_integration_report()
        else:  # all
            results = {}
            results['labels'] = integrations.sync_github_labels()
            results['boards'] = integrations.sync_project_boards()
            results['report'] = integrations.generate_integration_report()
            result = results
        
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