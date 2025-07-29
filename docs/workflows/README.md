# GitHub Actions Workflows

This directory contains documentation for recommended GitHub Actions workflows for the HF Eco2AI Plugin project.

## Available Workflow Templates

The following workflow templates are provided for manual setup in your `.github/workflows/` directory:

### Core Workflows

1. **[ci.yml](./ci.yml.template)** - Continuous Integration
   - Code quality checks (linting, formatting, type checking)
   - Multi-Python version testing
   - Security scanning
   - Coverage reporting

2. **[security.yml](./security.yml.template)** - Security Scanning
   - Dependency vulnerability scanning
   - SAST with CodeQL
   - Container security scanning
   - SBOM generation

3. **[release.yml](./release.yml.template)** - Release Automation
   - Automated PyPI publishing
   - GitHub releases with assets
   - Documentation deployment
   - Version tagging

4. **[docs.yml](./docs.yml.template)** - Documentation
   - Sphinx documentation building
   - GitHub Pages deployment
   - API documentation updates

### Specialized Workflows

5. **[performance.yml](./performance.yml.template)** - Performance Testing
   - Benchmark testing
   - Memory profiling
   - Performance regression detection

6. **[carbon-tracking.yml](./carbon-tracking.yml.template)** - Carbon Impact
   - Training carbon footprint measurement
   - Carbon budget enforcement
   - Environmental impact reporting

## Setup Instructions

1. Copy the desired workflow templates from this directory to `.github/workflows/`
2. Remove the `.template` extension from the filenames
3. Review and customize the workflows for your specific needs
4. Commit and push to enable the workflows

## Required Secrets

Configure these secrets in your GitHub repository settings:

- `PYPI_API_TOKEN` - PyPI publishing token
- `CODECOV_TOKEN` - Code coverage reporting
- `GPG_PRIVATE_KEY` - For signed releases (optional)

## Integration with Existing Tools

These workflows integrate with the existing development setup:
- Uses `Makefile` targets for consistency
- Respects `pyproject.toml` configuration
- Leverages existing pre-commit hooks
- Maintains compatibility with local development workflow

## Monitoring and Alerts

Configure GitHub repository settings for:
- Branch protection rules requiring status checks
- Required reviewers for pull requests
- Automatic security updates
- Dependency review

## Best Practices

- Always test workflows in a fork first
- Use matrix strategies for multi-environment testing
- Implement proper caching for faster builds
- Use workflow concurrency controls to prevent resource conflicts
- Regular security updates for action versions