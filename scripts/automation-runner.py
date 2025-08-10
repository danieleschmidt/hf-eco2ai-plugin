#!/usr/bin/env python3
"""Automation runner for scheduled tasks."""

import argparse
import logging
import subprocess
import sys
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_task(task_name: str):
    """Run automation task."""
    logger.info(f"Running task: {task_name}")
    
    tasks = {
        'metrics': ['python', 'scripts/collect-metrics.py', '--update'],
        'maintenance': ['python', 'scripts/maintenance.py', '--task', 'full'],
        'security': ['python', 'scripts/validate-setup.py', '--category', 'security'],
        'health': ['python', 'scripts/maintenance.py', '--task', 'health']
    }
    
    if task_name not in tasks:
        logger.error(f"Unknown task: {task_name}")
        return False
    
    try:
        result = subprocess.run(tasks[task_name], check=True)
        logger.info(f"Task {task_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Task {task_name} failed: {e}")
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['metrics', 'maintenance', 'security', 'health'])
    args = parser.parse_args()
    
    success = run_task(args.task)
    sys.exit(0 if success else 1)
