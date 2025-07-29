# GitHub Actions Workflows

This directory contains template workflows for the HF Eco2AI Plugin. Due to GitHub App permissions, these workflows need to be manually created by repository maintainers.

## Required Workflows

### 1. Test Workflow (`.github/workflows/test.yml`)

Comprehensive testing pipeline supporting:
- **Multi-Python Testing**: Python 3.10, 3.11, 3.12
- **Multi-OS Support**: Ubuntu, Windows, macOS
- **Test Types**: Unit, integration, performance tests
- **Coverage Reporting**: Codecov integration
- **Parallel Execution**: pytest-xdist for faster runs

**Key Features**:
- Automated dependency caching
- Test result artifacts upload
- Coverage threshold enforcement (80%)
- Performance regression detection

### 2. Security Workflow (`.github/workflows/security.yml`)

Enterprise-grade security scanning:
- **SAST Tools**: Bandit, Semgrep for static analysis
- **Dependency Scanning**: Safety, pip-audit for vulnerabilities
- **Secret Detection**: TruffleHog for credential scanning
- **Code Analysis**: CodeQL for comprehensive security review

**Security Gates**:
- Fail on moderate+ severity vulnerabilities
- Block commits with detected secrets
- Automated security report generation
- Integration with GitHub Security tab

### 3. Release Workflow (`.github/workflows/release.yml`)

Automated release engineering:
- **Semantic Versioning**: Automated version management
- **SBOM Generation**: Supply chain security with CycloneDX
- **Multi-Format Release**: GitHub releases, PyPI publishing
- **Security Validation**: Pre-release security scanning

**Release Process**:
- Trigger on version tags (v*.*.*)
- Generate comprehensive changelog
- Create signed releases with provenance
- Automated PyPI publishing with OIDC

## Manual Setup Instructions

### Step 1: Create Workflow Directory
```bash
mkdir -p .github/workflows
```

### Step 2: Copy Templates
```bash
cp docs/workflows/ci.yml.template .github/workflows/test.yml
cp docs/workflows/security.yml.template .github/workflows/security.yml
cp docs/workflows/release.yml.template .github/workflows/release.yml
```

### Step 3: Configure Repository Secrets
Add the following repository secrets in GitHub Settings → Secrets and variables → Actions:

#### Required Secrets
- `CODECOV_TOKEN` - For coverage reporting integration
- `PYPI_API_TOKEN` - For automated PyPI publishing (use OIDC if available)

#### Optional Secrets  
- `SLACK_WEBHOOK_URL` - For build notifications
- `DISCORD_WEBHOOK_URL` - For community notifications

### Step 4: Configure Environments

Create the following environments in GitHub Settings → Environments:

#### PyPI Environment
- **Name**: `pypi`
- **URL**: `https://pypi.org/p/hf-eco2ai-plugin`
- **Protection Rules**: Require reviewers for production releases
- **Secrets**: PYPI_API_TOKEN or enable OIDC trusted publishing

#### Test PyPI Environment
- **Name**: `testpypi`
- **URL**: `https://test.pypi.org/p/hf-eco2ai-plugin`
- **Protection Rules**: No restrictions for pre-releases
- **Secrets**: TEST_PYPI_API_TOKEN

### Step 5: Enable Branch Protection

Configure branch protection rules for `main` branch:
- **Required Status Checks**: `test`, `security-scan`, `lint`
- **Dismiss Stale Reviews**: Enabled
- **Require Branches Up-to-date**: Enabled
- **Restrict Pushes**: Admins and maintainers only
- **Allow Force Pushes**: Disabled

## Integration with Existing Tools

### Pre-commit Integration
The workflows integrate seamlessly with existing pre-commit hooks:
```yaml
# Validates workflow files in pre-commit
- repo: https://github.com/pre-commit/pre-commit-hooks
  rev: v5.0.0
  hooks:
    - id: check-yaml
      files: ^\.github/workflows/.*\.yml$
```

### Makefile Integration
Enhanced Makefile already includes commands that align with workflows:
- `make test` - Runs same tests as CI
- `make security` - Runs same security scans
- `make validate` - Comprehensive validation matching CI

### Docker Integration
Workflows leverage existing Docker configuration:
- Uses multi-stage Dockerfile for testing
- Integrates with docker-compose profiles
- Supports containerized testing environments

## Advanced Features

### Matrix Testing Configuration
```yaml
strategy:
  fail-fast: false
  matrix:
    os: [ubuntu-latest, windows-latest, macos-latest]
    python-version: ["3.10", "3.11", "3.12"]
    include:
      - os: ubuntu-latest
        python-version: "3.11"
        run-integration-tests: true
```

### Conditional Workflow Execution
```yaml
# Run security scan on schedule or security-related changes
on:
  schedule:
    - cron: '0 4 * * 2'  # Weekly Tuesday 4 AM
  push:
    paths:
      - 'src/**'
      - 'pyproject.toml'
      - '.github/workflows/security.yml'
```

### Artifact Management
```yaml
# Upload test results and coverage reports
- uses: actions/upload-artifact@v4
  if: always()
  with:
    name: test-results-${{ matrix.os }}-${{ matrix.python-version }}
    path: |
      junit.xml
      htmlcov/
    retention-days: 30
```

## Monitoring and Observability

### Status Badges
Add to README.md for visibility:
```markdown
[![Tests](https://github.com/terragonlabs/hf-eco2ai-plugin/workflows/Tests/badge.svg)](https://github.com/terragonlabs/hf-eco2ai-plugin/actions/workflows/test.yml)
[![Security](https://github.com/terragonlabs/hf-eco2ai-plugin/workflows/Security/badge.svg)](https://github.com/terragonlabs/hf-eco2ai-plugin/actions/workflows/security.yml)
[![Release](https://github.com/terragonlabs/hf-eco2ai-plugin/workflows/Release/badge.svg)](https://github.com/terragonlabs/hf-eco2ai-plugin/actions/workflows/release.yml)
```

### Performance Metrics
Target performance benchmarks:
- **Test Workflow**: <15 minutes total runtime
- **Security Workflow**: <10 minutes for complete scan
- **Release Workflow**: <20 minutes including publishing

### Notification Configuration
```yaml
# Slack notification on failure
- name: Notify on failure
  if: failure()  
  uses: 8398a7/action-slack@v3
  with:
    status: failure
    webhook_url: ${{ secrets.SLACK_WEBHOOK_URL }}
```

## Troubleshooting Guide

### Common Setup Issues

#### Permission Denied
```bash
# Error: refusing to allow a GitHub App to create workflows
# Solution: Repository admin must manually create workflow files
git checkout -b add-workflows
mkdir -p .github/workflows
cp docs/workflows/*.template .github/workflows/
# Remove .template extensions and commit
```

#### Secret Not Found
```bash
# Error: secret CODECOV_TOKEN not found
# Solution: Add secret in repository settings
# Settings → Secrets and variables → Actions → New repository secret
```

#### Workflow Not Triggering
```bash
# Check workflow syntax
yamllint .github/workflows/*.yml

# Verify trigger conditions
git log --oneline  # Check if commits match trigger paths
```

### Performance Optimization

#### Dependency Caching
```yaml
- uses: actions/cache@v4
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('pyproject.toml') }}
    restore-keys: |
      ${{ runner.os }}-pip-
```

#### Smart Test Selection
```yaml
# Run only affected tests
- name: Run affected tests
  run: |
    pytest --lf --co -q | grep -E "test_.*\.py" | \
    xargs pytest --maxfail=1 -v
```

#### Parallel Execution
```yaml
# Use all available cores
- name: Run tests in parallel
  run: |
    pytest -n auto --dist worksteal tests/
```

This comprehensive workflow documentation ensures that repository maintainers can easily implement enterprise-grade CI/CD, security scanning, and release automation while maintaining the existing development workflow and tooling.