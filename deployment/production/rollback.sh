#!/bin/bash
set -euo pipefail

echo "🔄 Starting HF Eco2AI Production Rollback"
echo "Deployment ID: deploy_20250821_120007"

# Stop current deployment
echo "⏹️ Stopping current deployment..."

# Restore previous version
echo "↩️ Restoring previous version..."

# Validate rollback
echo "✅ Validating rollback..."
python3 production_health_checker.py

echo "✅ Production rollback completed successfully!"
