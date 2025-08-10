# Production-ready multi-stage Dockerfile for HF Eco2AI Plugin
# Stage 1: Build dependencies and compile
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY README.md LICENSE ./

# Install the package
RUN pip install --no-cache-dir .

# Stage 2: Production runtime
FROM python:3.11-slim as production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get clean

# Create application user
RUN useradd --create-home --shell /bin/bash app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create application directories
RUN mkdir -p /app/data /app/logs /app/config && \
    chown -R app:app /app

# Set working directory and user
WORKDIR /app
USER app

# Copy configuration files
COPY --chown=app:app config/ ./config/
COPY --chown=app:app scripts/entrypoint.sh ./entrypoint.sh

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_ECO2AI_CONFIG_PATH=/app/config

# Entry point
ENTRYPOINT ["./entrypoint.sh"]
CMD ["hf-eco2ai", "--serve", "--host", "0.0.0.0", "--port", "8000"]

# Labels for metadata
LABEL maintainer="daniel@terragonlabs.com" \
      version="0.1.0" \
      description="HF Eco2AI Plugin - Production Container" \
      org.opencontainers.image.source="https://github.com/terragonlabs/hf-eco2ai-plugin"
