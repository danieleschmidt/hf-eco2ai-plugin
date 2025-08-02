# GitHub Actions Workflow Setup Guide

## Overview

This guide provides comprehensive instructions for setting up GitHub Actions workflows for the HF Eco2AI Plugin. Due to GitHub App permission limitations, workflows must be manually created by repository maintainers.

## Quick Setup (5 minutes)

### 1. Create Workflow Directory

```bash
mkdir -p .github/workflows
```

### 2. Copy Essential Workflows

```bash
# Copy the main CI/CD workflow
cp docs/workflows/comprehensive-ci.yml.template .github/workflows/ci.yml

# Copy security workflow
cp docs/workflows/security.yml.template .github/workflows/security.yml

# Copy release workflow
cp docs/workflows/release.yml.template .github/workflows/release.yml
```

### 3. Configure Repository Secrets

Go to **Settings** → **Secrets and variables** → **Actions** and add:

```
CODECOV_TOKEN=your_codecov_token
PYPI_API_TOKEN=your_pypi_token
SLACK_WEBHOOK_URL=your_slack_webhook (optional)
```

### 4. Commit and Push

```bash
git add .github/workflows/
git commit -m "ci: add GitHub Actions workflows"
git push
```

## Complete Setup Guide

### Workflow Templates Available

| Template | Purpose | Priority | Setup Time |
|----------|---------|----------|------------|
| `comprehensive-ci.yml.template` | Full CI/CD pipeline | ✅ Essential | 5 min |
| `security.yml.template` | Security scanning | ✅ Essential | 3 min |
| `release.yml.template` | Automated releases | ✅ Essential | 5 min |
| `carbon-tracking.yml.template` | Carbon footprint monitoring | ✨ Recommended | 10 min |
| `dependency-update.yml.template` | Automated dependency updates | ✨ Recommended | 15 min |

### Step-by-Step Setup

#### Step 1: Prepare Repository

```bash
# Ensure you're on the main branch
git checkout main
git pull origin main

# Create workflow directory
mkdir -p .github/workflows

# Verify templates exist
ls docs/workflows/*.template
```

#### Step 2: Configure Essential Workflows

##### A. Comprehensive CI/CD Pipeline

```bash
# Copy template
cp docs/workflows/comprehensive-ci.yml.template .github/workflows/ci.yml

# Review configuration (optional)
vim .github/workflows/ci.yml
```

**Key Features:**
- Multi-OS testing (Ubuntu, Windows, macOS)
- Multi-Python version support (3.10, 3.11, 3.12)
- Comprehensive test suite (unit, integration, e2e)
- Code coverage reporting
- Security scanning integration
- Automatic package building

##### B. Security Scanning

```bash
# Copy template
cp docs/workflows/security.yml.template .github/workflows/security.yml
```

**Security Tools Included:**
- Bandit (SAST)
- Safety (dependency vulnerabilities)
- pip-audit (package auditing)
- CodeQL (GitHub's semantic analysis)
- TruffleHog (secret detection)

##### C. Release Automation

```bash
# Copy template
cp docs/workflows/release.yml.template .github/workflows/release.yml
```

**Release Features:**
- Semantic versioning
- Automated changelog generation
- PyPI publishing with OIDC
- GitHub release creation
- SBOM generation

#### Step 3: Configure Advanced Workflows (Optional)

##### A. Carbon Tracking

```bash
# Copy template
cp docs/workflows/carbon-tracking.yml.template .github/workflows/carbon-tracking.yml
```

**Features:**
- CI/CD carbon footprint calculation
- Budget enforcement
- Sustainability reporting
- Optimization recommendations

**Additional Setup:**
```bash
# Add carbon tracking secrets
echo "SLACK_WEBHOOK_SUSTAINABILITY=your_sustainability_webhook" >> .env.secrets
```

##### B. Dependency Updates

```bash
# Copy template
cp docs/workflows/dependency-update.yml.template .github/workflows/dependency-update.yml
```

**Features:**
- Automated security updates
- Scheduled dependency updates
- Compatibility testing
- Auto-PR creation

#### Step 4: Repository Configuration

##### A. Required Secrets

In **GitHub Settings** → **Secrets and variables** → **Actions**:

| Secret Name | Purpose | Required | How to Get |
|-------------|---------|----------|------------|
| `CODECOV_TOKEN` | Code coverage reporting | Yes | [codecov.io](https://codecov.io) |
| `PYPI_API_TOKEN` | PyPI package publishing | Yes | [PyPI Account Settings](https://pypi.org/manage/account/) |
| `SLACK_WEBHOOK_URL` | Build notifications | No | [Slack Apps](https://api.slack.com/apps) |
| `SLACK_WEBHOOK_SUSTAINABILITY` | Carbon notifications | No | [Slack Apps](https://api.slack.com/apps) |

##### B. Environment Configuration

Create environments in **Settings** → **Environments**:

**PyPI Production Environment:**
```yaml
Name: pypi
URL: https://pypi.org/p/hf-eco2ai-plugin
Protection Rules:
  - Required reviewers: 1
  - Deployment branches: main only
Secrets:
  - PYPI_API_TOKEN
```

**Test PyPI Environment:**
```yaml
Name: testpypi
URL: https://test.pypi.org/p/hf-eco2ai-plugin
Protection Rules: None
Secrets:
  - TEST_PYPI_API_TOKEN
```

##### C. Branch Protection Rules

For the `main` branch in **Settings** → **Branches**:

```yaml
Protection Rules:
  - Require a pull request before merging: ✅
  - Require status checks to pass: ✅
    Required status checks:
      - Test (ubuntu-latest, 3.11, integration)
      - Pre-commit Checks
      - Security Scanning
      - Quality Gate
  - Dismiss stale PR approvals: ✅
  - Require branches to be up to date: ✅
  - Require conversation resolution: ✅
  - Restrict pushes that create files: ✅
  - Allow force pushes: ❌
  - Allow deletions: ❌
```

#### Step 5: Validation and Testing

##### A. Validate Workflow Syntax

```bash
# Install act (optional, for local testing)
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Validate workflow syntax
yamllint .github/workflows/*.yml

# Test workflows locally (optional)
act --list
act pull_request
```

##### B. Trigger Initial Workflow Run

```bash
# Commit and push workflows
git add .github/workflows/
git commit -m "ci: add comprehensive GitHub Actions workflows

- Add comprehensive CI/CD pipeline with multi-OS testing
- Add security scanning with multiple tools
- Add automated release workflow
- Add carbon tracking for sustainability monitoring
- Add automated dependency update workflow
"
git push origin main
```

##### C. Verify Workflow Execution

1. Go to **Actions** tab in GitHub
2. Verify workflows are running
3. Check for any failures and resolve
4. Confirm all status checks pass

#### Step 6: Add Status Badges (Optional)

Add to your `README.md`:

```markdown
## Build Status

[![CI/CD](https://github.com/terragonlabs/hf-eco2ai-plugin/workflows/Comprehensive%20CI%2FCD/badge.svg)](https://github.com/terragonlabs/hf-eco2ai-plugin/actions/workflows/ci.yml)
[![Security](https://github.com/terragonlabs/hf-eco2ai-plugin/workflows/Security/badge.svg)](https://github.com/terragonlabs/hf-eco2ai-plugin/actions/workflows/security.yml)
[![Release](https://github.com/terragonlabs/hf-eco2ai-plugin/workflows/Release/badge.svg)](https://github.com/terragonlabs/hf-eco2ai-plugin/actions/workflows/release.yml)
[![Carbon Tracking](https://github.com/terragonlabs/hf-eco2ai-plugin/workflows/Carbon%20Tracking/badge.svg)](https://github.com/terragonlabs/hf-eco2ai-plugin/actions/workflows/carbon-tracking.yml)

[![codecov](https://codecov.io/gh/terragonlabs/hf-eco2ai-plugin/branch/main/graph/badge.svg)](https://codecov.io/gh/terragonlabs/hf-eco2ai-plugin)
[![PyPI version](https://badge.fury.io/py/hf-eco2ai-plugin.svg)](https://badge.fury.io/py/hf-eco2ai-plugin)
[![Python versions](https://img.shields.io/pypi/pyversions/hf-eco2ai-plugin.svg)](https://pypi.org/project/hf-eco2ai-plugin/)
```

## Workflow Customization

### Environment Variables

Customize workflows by modifying environment variables at the top of each file:

```yaml
# In ci.yml
env:
  PYTHON_DEFAULT: '3.11'  # Change default Python version
  POETRY_VERSION: '1.7.1'  # Update Poetry version
  PIP_CACHE_DIR: /tmp/pip-cache  # Cache directory
```

### Matrix Strategy Customization

```yaml
# Customize test matrix
strategy:
  matrix:
    os: [ubuntu-latest, windows-latest, macos-latest]
    python-version: ['3.10', '3.11', '3.12']
    # Add custom combinations
    include:
      - os: ubuntu-latest
        python-version: '3.11'
        run-integration-tests: true
        run-performance-tests: true
```

### Conditional Execution

```yaml
# Run only on specific paths
on:
  push:
    paths:
      - 'src/**'
      - 'tests/**'
      - 'pyproject.toml'
  pull_request:
    paths:
      - 'src/**'
      - 'tests/**'
```

### Schedule Customization

```yaml
# Customize schedule
on:
  schedule:
    - cron: '0 6 * * 1'  # Weekly Monday 6 AM
    - cron: '0 2 * * *'   # Daily 2 AM (for dependency updates)
```

## Troubleshooting

### Common Issues

#### 1. Permission Denied Error

**Error:** `refusing to allow a GitHub App to create or update workflow`

**Solution:**
```bash
# Repository admin must manually create workflows
git checkout -b add-workflows
mkdir -p .github/workflows
cp docs/workflows/*.template .github/workflows/
# Remove .template extensions
for f in .github/workflows/*.template; do mv "$f" "${f%.template}"; done
git add .github/workflows/
git commit -m "ci: add GitHub Actions workflows"
git push origin add-workflows
# Create PR and merge
```

#### 2. Secret Not Found

**Error:** `secret CODECOV_TOKEN not found`

**Solution:**
1. Go to repository **Settings**
2. Click **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add the required secret

#### 3. Workflow Not Triggering

**Possible Causes:**
- File in wrong location
- YAML syntax error
- Trigger conditions not met

**Debug Steps:**
```bash
# Check file location
ls -la .github/workflows/

# Validate YAML syntax
yamllint .github/workflows/*.yml

# Check trigger conditions
git log --oneline -5  # Recent commits
git show --name-only  # Files in last commit
```

#### 4. Test Failures

**Common Issues:**
- Missing test dependencies
- Environment differences
- Flaky tests

**Solutions:**
```yaml
# Add retry mechanism
- name: Run tests with retry
  uses: nick-invision/retry@v2
  with:
    timeout_minutes: 10
    max_attempts: 3
    command: pytest tests/ -v

# Use specific Python version
- name: Set up Python
  uses: actions/setup-python@v5
  with:
    python-version: '3.11.7'  # Specific version
```

### Performance Optimization

#### 1. Dependency Caching

```yaml
- name: Cache dependencies
  uses: actions/cache@v4
  with:
    path: |
      ~/.cache/pip
      ~/.cache/pre-commit
    key: ${{ runner.os }}-deps-${{ hashFiles('pyproject.toml', '.pre-commit-config.yaml') }}
    restore-keys: |
      ${{ runner.os }}-deps-
```

#### 2. Parallel Execution

```yaml
- name: Run tests in parallel
  run: |
    pytest -n auto --dist worksteal tests/
```

#### 3. Selective Testing

```yaml
# Only run affected tests
- name: Get changed files
  id: files
  run: |
    echo "files=$(git diff --name-only ${{ github.event.before }}..${{ github.sha }} | tr '\n' ' ')" >> $GITHUB_OUTPUT

- name: Run affected tests
  if: contains(steps.files.outputs.files, 'src/')
  run: pytest tests/
```

## Monitoring and Maintenance

### Workflow Analytics

1. **Actions tab** → **All workflows**
2. Monitor success rates
3. Identify bottlenecks
4. Track resource usage

### Regular Maintenance Tasks

```bash
# Monthly workflow review
# 1. Update action versions
sed -i 's/actions\/checkout@v3/actions\/checkout@v4/g' .github/workflows/*.yml

# 2. Review and update Python versions
# 3. Check for deprecated features
# 4. Optimize performance
# 5. Update secrets if needed
```

### Success Metrics

- **Build Success Rate:** >95%
- **Average Build Time:** <15 minutes
- **Security Scan Coverage:** 100%
- **Deployment Success Rate:** >99%
- **Carbon Efficiency:** <0.1 kg CO₂ per build

## Support and Resources

- **GitHub Actions Documentation:** https://docs.github.com/actions
- **Workflow Syntax:** https://docs.github.com/actions/reference/workflow-syntax-for-github-actions
- **Marketplace:** https://github.com/marketplace?type=actions
- **Community Forum:** https://github.community/c/code-to-cloud/github-actions
- **HF Eco2AI Plugin Issues:** https://github.com/terragonlabs/hf-eco2ai-plugin/issues

---

**Last Updated:** 2025-08-02  
**Version:** 1.0  
**Next Review:** 2025-11-02
