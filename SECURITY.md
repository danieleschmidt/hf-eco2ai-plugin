# Security Policy

## Supported Versions

We provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

### Private Reporting
- **Email**: security@terragonlabs.com
- **Subject**: [SECURITY] HF Eco2AI Plugin Vulnerability
- **Response time**: Within 48 hours

### What to Include
- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Suggested mitigation (if any)

### Our Process
1. **Acknowledgment** within 48 hours
2. **Initial assessment** within 1 week
3. **Fix development** and testing
4. **Coordinated disclosure** after fix is ready

## Security Considerations

### Data Handling
- Energy metrics are collected locally by default
- No sensitive training data is transmitted
- Carbon data comes from public grid intensity APIs

### Prometheus Export
- Metrics endpoint requires explicit enabling
- Consider network security when exposing metrics
- Use authentication for production deployments

### Dependencies
- Regular security scanning via `bandit` and `safety`
- Automated dependency updates via Dependabot
- Pin known-secure versions in production

### Best Practices
- Keep dependencies updated
- Use virtual environments
- Validate configuration inputs
- Enable audit logging for production use

## Security Features

- Input validation for all configuration parameters
- Safe file handling with proper permissions
- No eval() or exec() usage
- Minimal privilege requirements