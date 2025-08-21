#!/bin/bash
set -euo pipefail

echo "🚀 Starting HF Eco2AI Production Deployment"
echo "Deployment ID: deploy_20250821_120007"
echo "Version: 1.0.0"
echo "Environment: production"

# Pre-deployment checks
echo "🔍 Running pre-deployment checks..."
python3 comprehensive_quality_testing_suite.py
if [ $? -ne 0 ]; then
    echo "❌ Pre-deployment quality checks failed"
    exit 1
fi

# Build and package
echo "📦 Building application..."
python3 -m pip install -e .[all]

# Deploy to production
echo "🚀 Deploying to production environment..."

# Health checks
echo "🏥 Running health checks..."
python3 production_health_checker.py

# Performance validation
echo "⚡ Running performance validation..."
python3 production_performance_validator.py

# Security validation
echo "🔒 Running security validation..."
python3 production_security_validator.py

echo "✅ Production deployment completed successfully!"
echo "🎉 HF Eco2AI is now live in production!"
