#!/bin/bash
set -e

# Production entrypoint script for HF Eco2AI Plugin

echo "🚀 Starting HF Eco2AI Plugin in production mode..."

# Environment validation
if [ -z "$HF_ECO2AI_ENV" ]; then
    echo "ERROR: HF_ECO2AI_ENV environment variable not set"
    exit 1
fi

if [ "$HF_ECO2AI_ENV" != "production" ] && [ "$HF_ECO2AI_ENV" != "staging" ] && [ "$HF_ECO2AI_ENV" != "development" ]; then
    echo "ERROR: Invalid HF_ECO2AI_ENV value: $HF_ECO2AI_ENV"
    exit 1
fi

# Create necessary directories
mkdir -p /app/data /app/logs /app/config

# Set appropriate permissions
chmod 755 /app/data /app/logs
chmod 750 /app/config

# Validate configuration files
if [ ! -f "/app/config/config.json" ]; then
    echo "WARNING: No config.json found, using defaults"
    cat > /app/config/config.json << EOF
{
  "project_name": "hf-eco2ai-$HF_ECO2AI_ENV",
  "environment": "$HF_ECO2AI_ENV",
  "logging": {
    "level": "INFO",
    "format": "json"
  },
  "monitoring": {
    "enabled": true,
    "prometheus_port": 9091
  }
}
EOF
fi

# Initialize logging
exec > >(tee -a /app/logs/application.log)
exec 2>&1

echo "✅ Environment: $HF_ECO2AI_ENV"
echo "✅ Configuration validated"
echo "✅ Directories initialized"

# Health check endpoint setup
if [ "$1" = "hf-eco2ai" ] && [ "$2" = "--serve" ]; then
    echo "🌐 Starting web service..."
    
    # Background health check server
    cat > /tmp/health_server.py << 'EOF'
import http.server
import socketserver
import threading
import json
import time

class HealthHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            health_data = {
                "status": "healthy",
                "timestamp": time.time(),
                "version": "0.1.0",
                "environment": "production"
            }
            self.wfile.write(json.dumps(health_data).encode())
        elif self.path == '/ready':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            ready_data = {
                "ready": True,
                "timestamp": time.time()
            }
            self.wfile.write(json.dumps(ready_data).encode())
        else:
            self.send_response(404)
            self.end_headers()

def start_health_server():
    with socketserver.TCPServer(("", 8000), HealthHandler) as httpd:
        httpd.serve_forever()

if __name__ == "__main__":
    health_thread = threading.Thread(target=start_health_server, daemon=True)
    health_thread.start()
    print("Health server started on port 8000")
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
EOF

    python3 /tmp/health_server.py &
    HEALTH_PID=$!
    
    echo "✅ Health endpoints available at :8000/health and :8000/ready"
fi

# Signal handling for graceful shutdown
cleanup() {
    echo "🔄 Graceful shutdown initiated..."
    if [ ! -z "$HEALTH_PID" ]; then
        kill $HEALTH_PID 2>/dev/null || true
    fi
    exit 0
}

trap cleanup SIGTERM SIGINT

# Execute the main command
echo "🚀 Executing: $@"
exec "$@"
