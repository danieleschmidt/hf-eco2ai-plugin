# GitHub Actions Workflows

This document describes the recommended CI/CD workflows for the HF Eco2AI Plugin project.

## Overview

Since this project uses Terragon's policy of not modifying GitHub Actions files directly, this directory contains documentation and templates for the recommended workflows.

## Recommended Workflows

### 1. CI/CD Pipeline (`ci.yml`)

**Purpose**: Continuous integration for pull requests and main branch pushes.

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
        
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
        
    - name: Run pre-commit
      run: pre-commit run --all-files
      
    - name: Run tests
      run: pytest --cov=hf_eco2ai --cov-report=xml
      
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

### 2. Security Scanning (`security.yml`)

**Purpose**: Security vulnerability scanning and dependency checks.

```yaml
name: Security Scan

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Monday

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Run Bandit Security Scan
      uses: securecodewarrior/github-action-bandit@v1
      with:
        config_file: 'pyproject.toml'
        
    - name: Run Safety Check
      run: |
        pip install safety
        safety check --json
```

### 3. Release Automation (`release.yml`)

**Purpose**: Automated releases to PyPI when tags are pushed.

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
        
    - name: Build package
      run: |
        python -m pip install --upgrade pip build
        python -m build
        
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        password: ${{ secrets.PYPI_API_TOKEN }}
```

### 4. Documentation (`docs.yml`)

**Purpose**: Build and deploy documentation.

```yaml
name: Documentation

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  docs:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
        
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
        
    - name: Build docs
      run: |
        cd docs
        make html
        
    - name: Deploy to GitHub Pages
      if: github.ref == 'refs/heads/main'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs/_build/html
```

## Required Secrets

To use these workflows, configure the following secrets in your GitHub repository:

- `PYPI_API_TOKEN`: Token for publishing to PyPI
- `CODECOV_TOKEN`: Token for code coverage reporting (optional)

## Workflow Features

### Pre-commit Integration
All workflows run pre-commit hooks to ensure code quality:
- Black code formatting
- Ruff linting
- MyPy type checking
- Bandit security scanning
- Safety dependency checking

### Multi-Python Testing
CI tests against Python 3.10, 3.11, and 3.12 to ensure compatibility.

### Security Scanning
Automated security scanning includes:
- Bandit for common security issues
- Safety for known vulnerabilities in dependencies
- Dependabot for dependency updates

### Code Coverage
Integration with Codecov for tracking test coverage over time.

## Manual Setup Instructions

1. Copy the workflow templates from this directory to `.github/workflows/`
2. Configure the required secrets in your repository settings
3. Adjust the Python versions and dependencies as needed
4. Enable GitHub Pages if using documentation deployment

## Best Practices

1. **Branch Protection**: Configure branch protection rules requiring CI checks
2. **Review Requirements**: Require code review before merging
3. **Automated Testing**: Ensure all critical paths are covered by tests
4. **Security First**: Run security scans on every PR
5. **Documentation**: Keep documentation in sync with code changes

## Monitoring

Monitor workflow runs in the Actions tab of your GitHub repository. Failed runs should be investigated and fixed promptly to maintain code quality.