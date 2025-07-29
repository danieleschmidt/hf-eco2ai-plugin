# Contributing to HF Eco2AI Plugin

We welcome contributions! This guide outlines how to contribute effectively.

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/hf-eco2ai-plugin.git
   cd hf-eco2ai-plugin
   ```

2. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

## Development Workflow

### Code Quality
- Use `black` for formatting: `black src tests`
- Use `ruff` for linting: `ruff check src tests`
- Use `mypy` for type checking: `mypy src`
- All checks run automatically via pre-commit

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=hf_eco2ai --cov-report=html

# Run specific test types
pytest -m "not slow"  # Skip slow tests
pytest -m integration  # Only integration tests
```

### Documentation
- Build docs: `cd docs && make html`
- View docs: Open `docs/_build/html/index.html`

## Contribution Guidelines

### Pull Requests
1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes with tests
3. Ensure all checks pass: `pre-commit run --all-files`
4. Submit PR with clear description

### Commit Messages
- Use conventional commits: `feat:`, `fix:`, `docs:`, `test:`
- Be descriptive and concise
- Reference issues: `fixes #123`

### Code Standards
- Follow PEP 8 via `black` and `ruff`
- Add type hints for all public APIs
- Write docstrings for public functions
- Maintain test coverage >90%

## Priority Areas

1. **Energy Monitoring**: More accurate GPU/CPU power measurement
2. **Cloud Integrations**: AWS, GCP, Azure carbon data APIs
3. **Framework Support**: JAX, TensorFlow, LightGBM callbacks
4. **Visualization**: Enhanced Grafana dashboards
5. **Documentation**: Tutorials and best practices

## Reporting Issues

Use GitHub Issues with:
- Clear description of the problem
- Environment details (Python, transformers, GPU)
- Minimal code example
- Expected vs actual behavior

## License

By contributing, you agree your contributions will be licensed under MIT License.

## Questions?

- GitHub Discussions for general questions
- Discord: [Green AI Community](https://discord.gg/green-ai)
- Email: hf-eco2ai@terragonlabs.com