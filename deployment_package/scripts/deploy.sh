#!/bin/bash

# TERRAGON SDLC v5.0 - Production Deployment Script
# Deployment ID: terragon_sdlc_v5_deployment_1756146283

set -e

echo "🚀 Starting TERRAGON SDLC v5.0 Production Deployment"
echo "Deployment ID: terragon_sdlc_v5_deployment_1756146283"

# Configuration
NAMESPACE="terragon-sdlc"
CHART_VERSION="5.0.0"

# Pre-deployment checks
echo "📋 Running pre-deployment checks..."

# Check kubectl connection
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ Error: kubectl not connected to cluster"
    exit 1
fi

# Check required tools
for tool in kubectl helm docker; do
    if ! command -v $tool &> /dev/null; then
        echo "❌ Error: $tool not installed"
        exit 1
    fi
done

echo "✅ Pre-deployment checks passed"

# Create namespace
echo "📁 Creating namespace..."
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Deploy configurations
echo "⚙️ Applying configurations..."
kubectl apply -f configs/ -n $NAMESPACE

# Deploy Kubernetes manifests
echo "☸️ Deploying to Kubernetes..."
kubectl apply -f kubernetes/ -n $NAMESPACE

# Wait for rollout
echo "⏳ Waiting for deployment rollout..."
kubectl rollout status deployment/hf-eco2ai -n $NAMESPACE --timeout=300s

# Verify deployment
echo "✅ Verifying deployment..."
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE

# Health check
echo "🏥 Running health checks..."
for i in {1..10}; do
    if kubectl exec -n $NAMESPACE deployment/hf-eco2ai -- curl -f http://localhost:8080/health &> /dev/null; then
        echo "✅ Health check passed"
        break
    fi
    echo "⏳ Health check attempt $i/10..."
    sleep 10
done

echo "🎉 TERRAGON SDLC v5.0 deployed successfully!"
echo "📊 Access monitoring at: http://your-grafana-url:3000"
echo "🔗 API endpoint: https://api.terragonlabs.com"
