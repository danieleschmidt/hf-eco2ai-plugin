# GitHub Actions Workflow Setup Guide

## Overview
This repository has been enhanced with enterprise-grade GitHub Actions workflows that require manual setup due to GitHub security restrictions.

## Required Workflows
The following workflow files have been prepared in `docs/workflows/production/` and need to be manually copied to `.github/workflows/`:

### 1. Core CI/CD (`ci.yml`)
- **Purpose**: Multi-platform continuous integration 
- **Features**: Python 3.10-3.12 matrix testing, coverage reporting
- **Triggers**: Push to main/develop, PRs, weekly schedule
- **Dependencies**: Codecov token (optional)

### 2. Security Scanning (`security.yml`) 
- **Purpose**: Comprehensive security analysis
- **Features**: Bandit, Safety, pip-audit scanning with SARIF upload
- **Triggers**: Push to main, PRs, weekly schedule
- **Dependencies**: GitHub Advanced Security (for SARIF upload)

### 3. Release Automation (`release.yml`)
- **Purpose**: Automated PyPI publishing and GitHub releases
- **Features**: Build verification, PyPI publishing with attestations
- **Triggers**: Git tags (v*)
- **Dependencies**: PyPI token, release environment

### 4. Quality Gates (`quality.yml`)
- **Purpose**: Code quality enforcement
- **Features**: Formatting, linting, type checking, coverage thresholds
- **Triggers**: Push to main/develop, PRs
- **Dependencies**: None

## Setup Instructions

### Step 1: Copy Workflow Files
```bash
# Navigate to repository root
cd /path/to/hf-eco2ai-plugin

# Create workflows directory
mkdir -p .github/workflows

# Copy prepared workflows from documentation
cp docs/workflows/production/*.yml .github/workflows/
```

### Step 2: Configure Secrets
Add the following secrets in GitHub Settings > Secrets and variables > Actions:

- `CODECOV_TOKEN`: For coverage reporting (optional)
- `PYPI_API_TOKEN`: For automated releases (required for releases)

### Step 3: Enable GitHub Advanced Security (Optional)
For security SARIF uploads:
1. Go to Settings > Code security and analysis
2. Enable "Code scanning" 
3. Enable "Secret scanning"

### Step 4: Create Release Environment
For secure releases:
1. Go to Settings > Environments
2. Create environment named "release"
3. Add protection rules as needed
4. Add PyPI token to environment secrets

### Step 5: Configure Branch Protection
Recommended protection rules for `main` branch:
- Require status checks: CI, Security, Quality Gates
- Require branches to be up to date
- Require review from code owners
- Dismiss stale reviews

## Workflow Features

### Multi-Platform Support
- **Platforms**: Ubuntu, Windows, macOS
- **Python Versions**: 3.10, 3.11, 3.12
- **Test Coverage**: 85% minimum threshold

### Security Integration  
- **Static Analysis**: Bandit for Python security issues
- **Dependency Scanning**: Safety + pip-audit for vulnerabilities
- **Results Integration**: SARIF upload to GitHub Security tab

### Release Automation
- **Trigger**: Git tags matching `v*` pattern
- **Build Verification**: Package building and validation
- **PyPI Publishing**: Automated with attestations
- **GitHub Releases**: Auto-generated with artifacts

### Quality Enforcement
- **Code Formatting**: Black (88 char line length)
- **Linting**: Ruff with comprehensive rule set
- **Type Checking**: MyPy with strict configuration
- **Complexity Analysis**: Radon for maintainability metrics

## Validation

After setup, test the workflows:

1. **CI Test**: Create a PR with a small change
2. **Security Test**: Workflows run automatically  
3. **Quality Test**: Workflows enforce standards
4. **Release Test**: Create a test tag (e.g., `v0.1.0-test`)

## Integration with Existing Tools

These workflows integrate seamlessly with existing repository infrastructure:
- **Pre-commit hooks**: Local development quality
- **Dependabot**: Automated dependency updates  
- **Devcontainer**: Consistent development environment
- **Documentation**: Sphinx docs with GitHub Pages
- **Monitoring**: Prometheus/Grafana dashboards

## Rollback Procedure

If issues arise:
1. Disable problematic workflow in `.github/workflows/`
2. Remove workflow file temporarily
3. Fix configuration issues
4. Re-enable workflow

All workflows are designed to fail gracefully and provide clear error messages for troubleshooting.