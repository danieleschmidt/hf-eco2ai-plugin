# GitHub Copilot Instructions for HF Eco2AI Plugin

## Project Context
You are working on a Hugging Face Transformers callback plugin for carbon emission tracking using Eco2AI. This is a production-ready ML sustainability tool with enterprise-grade requirements.

## Code Standards
- **Python 3.10+** with full type annotations
- **Async-first** where applicable for better performance
- **Error handling** with custom exceptions and proper logging
- **Security-conscious** - never log sensitive data
- **Performance-optimized** - minimize overhead during training

## Architecture Patterns
- **Callback Pattern**: Extend `TrainerCallback` for HF integration  
- **Plugin Architecture**: Modular design with optional features
- **Configuration-driven**: Use dataclasses for settings
- **Observer Pattern**: For metric collection and reporting

## ML-Specific Guidelines
- **Minimal Training Impact**: Carbon tracking should add <1% overhead
- **GPU-Aware**: Handle multi-GPU setups gracefully
- **Framework Agnostic**: Support both HF Transformers and PyTorch Lightning
- **Real-time Metrics**: Provide live carbon tracking during training

## Testing Requirements
- **Mock External APIs**: Never call real carbon APIs in tests
- **GPU Tests**: Mark with `@pytest.mark.gpu` decorator
- **Performance Tests**: Use `pytest-benchmark` for overhead testing
- **Contract Tests**: Verify API compatibility with major ML frameworks

## Documentation Style
- **Google Docstrings** for all public APIs
- **Type hints** are mandatory, not optional
- **Examples** should be copy-pastable and work immediately
- **Performance notes** for any methods that might impact training

## Security Considerations
- **No API keys in logs** - sanitize all output
- **Validate inputs** from external carbon APIs
- **Rate limiting** for external API calls
- **Secure defaults** - fail safe when carbon tracking fails

## Suggestions Priority
1. **Correctness** - Code must work reliably during ML training
2. **Performance** - Minimize impact on training speed
3. **Security** - Protect sensitive training data and API keys  
4. **Maintainability** - Clear, well-documented code
5. **Innovation** - Suggest modern Python patterns and ML best practices