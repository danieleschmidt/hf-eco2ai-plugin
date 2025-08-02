# Multi-stage build for HF Eco2AI Plugin
FROM python:3.13-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Add metadata labels
LABEL org.opencontainers.image.title="HF Eco2AI Plugin"
LABEL org.opencontainers.image.description="Hugging Face Trainer callback for CO₂ tracking with Eco2AI"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL org.opencontainers.image.source="https://github.com/terragonlabs/hf-eco2ai-plugin"
LABEL org.opencontainers.image.licenses="MIT"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml ./
COPY README.md ./
COPY LICENSE ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir build wheel

# Copy source code
COPY src/ src/

# Build the package
RUN python -m build --wheel --no-isolation

# Production stage
FROM python:3.13-slim as production

# Create non-root user
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy built wheel from builder stage
COPY --from=builder /app/dist/*.whl ./

# Install the package
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir *.whl \
    && rm -f *.whl

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import hf_eco2ai; print('OK')" || exit 1

# Default command
CMD ["python", "-c", "import hf_eco2ai; print('HF Eco2AI Plugin is ready!')"]

# Development stage
FROM production as development

# Switch back to root for package installation
USER root

# Install development dependencies
RUN pip install --no-cache-dir pytest pytest-cov black ruff mypy pre-commit

# Install additional dev tools
RUN apt-get update && apt-get install -y \
    vim \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Switch back to appuser
USER appuser

CMD ["bash"]