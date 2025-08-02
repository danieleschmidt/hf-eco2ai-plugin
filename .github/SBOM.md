# Software Bill of Materials (SBOM)

## Project Information

- **Project Name**: HF Eco2AI Plugin
- **Version**: 0.1.0
- **License**: MIT
- **Repository**: https://github.com/terragonlabs/hf-eco2ai-plugin

## Core Dependencies

### Runtime Dependencies

| Package | Version | License | Purpose |
|---------|---------|---------|----------|
| transformers | >=4.40.0 | Apache-2.0 | Hugging Face Transformers integration |
| torch | >=2.0.0 | BSD-3-Clause | PyTorch deep learning framework |
| eco2ai | >=2.0.0 | Apache-2.0 | Energy consumption tracking |
| pynvml | >=11.5.0 | BSD-3-Clause | NVIDIA GPU monitoring |
| prometheus-client | >=0.20.0 | Apache-2.0 | Metrics export to Prometheus |
| pandas | >=2.0.0 | BSD-3-Clause | Data manipulation and analysis |
| plotly | >=5.20.0 | MIT | Interactive visualization |
| carbontracker | >=1.5.0 | MIT | Carbon footprint tracking |
| codecarbon | >=2.3.0 | MIT | ML carbon emissions tracking |

### Optional Dependencies

#### Lightning Integration
| Package | Version | License | Purpose |
|---------|---------|---------|----------|
| pytorch-lightning | >=2.2.0 | Apache-2.0 | PyTorch Lightning integration |

#### MLflow Integration
| Package | Version | License | Purpose |
|---------|---------|---------|----------|
| mlflow | >=2.0.0 | Apache-2.0 | Experiment tracking |

#### Grafana Integration
| Package | Version | License | Purpose |
|---------|---------|---------|----------|
| grafana-api | >=1.0.3 | MIT | Grafana dashboard management |

#### AI-Ops Integration
| Package | Version | License | Purpose |
|---------|---------|---------|----------|
| opentelemetry-api | >=1.20.0 | Apache-2.0 | Observability framework |
| opentelemetry-sdk | >=1.20.0 | Apache-2.0 | OpenTelemetry SDK |
| opentelemetry-instrumentation-requests | >=0.41b0 | Apache-2.0 | HTTP request instrumentation |
| structlog | >=23.0.0 | Apache-2.0/MIT | Structured logging |
| rich | >=13.0.0 | MIT | Rich text and beautiful formatting |

#### Performance Optimization
| Package | Version | License | Purpose |
|---------|---------|---------|----------|
| numba | >=0.58.0 | BSD-2-Clause | JIT compilation for performance |
| asyncio-throttle | >=1.0.2 | MIT | Asynchronous request throttling |
| aiohttp | >=3.9.0 | Apache-2.0 | Asynchronous HTTP client/server |
| orjson | >=3.9.0 | Apache-2.0/MIT | Fast JSON serialization |

### Development Dependencies

#### Testing
| Package | Version | License | Purpose |
|---------|---------|---------|----------|
| pytest | >=7.0.0 | MIT | Testing framework |
| pytest-cov | >=4.0.0 | MIT | Coverage plugin for pytest |
| pytest-mock | >=3.10.0 | MIT | Mock plugin for pytest |
| pytest-benchmark | >=4.0.0 | BSD-2-Clause | Benchmarking plugin for pytest |
| pytest-xdist | >=3.0.0 | MIT | Distributed testing |

#### Code Quality
| Package | Version | License | Purpose |
|---------|---------|---------|----------|
| black | >=23.0.0 | MIT | Code formatting |
| ruff | >=0.1.0 | MIT | Fast Python linter |
| mypy | >=1.5.0 | MIT | Static type checking |
| pre-commit | >=3.0.0 | MIT | Pre-commit hooks framework |

#### Documentation
| Package | Version | License | Purpose |
|---------|---------|---------|----------|
| sphinx | >=6.0.0 | BSD-2-Clause | Documentation generation |
| sphinx-rtd-theme | >=1.3.0 | MIT | Read the Docs theme for Sphinx |

#### Security
| Package | Version | License | Purpose |
|---------|---------|---------|----------|
| bandit | >=1.7.5 | Apache-2.0 | Security linting |
| safety | >=2.3.0 | MIT | Dependency vulnerability checking |
| pip-audit | >=2.6.0 | Apache-2.0 | Audit Python packages for vulnerabilities |

#### Release Management
| Package | Version | License | Purpose |
|---------|---------|---------|----------|
| mutmut | >=2.4.0 | MIT | Mutation testing |
| cyclonedx-bom | >=4.0.0 | Apache-2.0 | SBOM generation |
| git-cliff | >=1.4.0 | Apache-2.0/MIT | Changelog generation |

## System Dependencies

### Runtime System Dependencies
| Component | Version | Purpose |
|-----------|---------|----------|
| Python | >=3.10 | Runtime environment |
| CUDA | >=11.0 (optional) | GPU acceleration |
| NVIDIA Driver | >=470.0 (optional) | GPU support |

### Build System Dependencies
| Component | Version | Purpose |
|-----------|---------|----------|
| setuptools | >=61.0 | Python packaging |
| wheel | latest | Wheel building |
| build | latest | Build frontend |

## Container Dependencies

### Base Images
| Image | Version | Purpose |
|-------|---------|----------|
| python:3.10-slim | latest | Production runtime |
| prom/prometheus | v2.45.0 | Metrics collection |
| grafana/grafana | 10.0.0 | Visualization dashboards |

### Container System Packages
| Package | Purpose |
|---------|----------|
| gcc | C compiler for native extensions |
| g++ | C++ compiler |
| git | Version control |

## Security Considerations

### Known Vulnerabilities
- No known high or critical vulnerabilities as of last scan
- Regular security scanning performed with bandit, safety, and pip-audit
- Container images scanned with Trivy

### Supply Chain Security
- All dependencies pinned to minimum versions
- Package integrity verified through pip's hash checking
- Container images built from official base images
- Regular dependency updates and security patches

### License Compliance
- All dependencies use OSI-approved open source licenses
- No GPL or copyleft licenses that would affect distribution
- Primarily MIT, Apache-2.0, and BSD licenses

## Generation Information

- **Generated**: 2025-08-02
- **Generator**: Manual compilation with automated verification
- **Format**: CycloneDX JSON (available separately)
- **Verification**: pip-licenses, cyclonedx-bom

## Maintenance

This SBOM is updated with each release and whenever dependencies are modified. For the most current information, see:

- `pyproject.toml` for Python dependencies
- `Dockerfile` for container dependencies
- CI/CD pipelines for build-time dependencies

For automated SBOM generation in various formats:

```bash
# Generate CycloneDX JSON format
cyclonedx-py -o sbom.json

# Generate SPDX format
cyclonedx-py -o sbom.spdx --format spdx

# Generate dependency tree
pipdeptree --json-tree > dependency-tree.json
```
