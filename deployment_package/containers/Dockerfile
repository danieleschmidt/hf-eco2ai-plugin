# TERRAGON SDLC v5.0 - Production Dockerfile
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create build directory
WORKDIR /build

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash terragon

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /home/terragon/.local

# Set up application directory
WORKDIR /app
RUN chown terragon:terragon /app

# Copy application code
COPY --chown=terragon:terragon . .

# Switch to non-root user
USER terragon

# Set environment variables
ENV PATH=/home/terragon/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV TERRAGON_ENV=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Start application
CMD ["python", "-m", "hf_eco2ai.api"]
