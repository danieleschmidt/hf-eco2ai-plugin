#!/bin/bash
# Build script for HF Eco2AI Plugin

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
BUILD_TYPE="production"
PUSH_IMAGES=false
RUN_TESTS=true
CLEAN_BUILD=false
VERBOSE=false

# Get build metadata
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
VERSION=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/' || echo "0.1.0")
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build HF Eco2AI Plugin Docker images and packages

OPTIONS:
    -t, --type TYPE         Build type: production, development, or all (default: production)
    -p, --push              Push images to registry after build
    -T, --no-tests          Skip running tests
    -c, --clean             Clean build (remove existing images and cache)
    -v, --verbose           Verbose output
    -h, --help              Show this help message

EXAMPLES:
    $0                      # Build production image
    $0 -t development       # Build development image
    $0 -t all -p            # Build all images and push to registry
    $0 -c -v                # Clean build with verbose output

EOF
}

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✓${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ✗${NC} $1" >&2
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -p|--push)
            PUSH_IMAGES=true
            shift
            ;;
        -T|--no-tests)
            RUN_TESTS=false
            shift
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Validate build type
if [[ ! "$BUILD_TYPE" =~ ^(production|development|all)$ ]]; then
    error "Invalid build type: $BUILD_TYPE"
    print_usage
    exit 1
fi

# Set verbose mode
if [[ "$VERBOSE" == "true" ]]; then
    set -x
fi

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
        exit 1
    fi
    
    success "Prerequisites check passed"
}

# Clean build artifacts
clean_build_artifacts() {
    if [[ "$CLEAN_BUILD" == "true" ]]; then
        log "Cleaning build artifacts..."
        
        # Remove Python build artifacts
        rm -rf build/ dist/ *.egg-info/
        find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
        find . -name "*.pyc" -delete 2>/dev/null || true
        
        # Remove Docker images
        docker image prune -f
        docker images | grep "hf-eco2ai-plugin" | awk '{print $3}' | xargs -r docker rmi -f
        
        success "Build artifacts cleaned"
    fi
}

# Build Python package
build_python_package() {
    log "Building Python package..."
    
    # Install build dependencies
    python -m pip install --upgrade pip build wheel
    
    # Build package
    python -m build --wheel --sdist
    
    success "Python package built successfully"
}

# Run tests
run_tests() {
    if [[ "$RUN_TESTS" == "true" ]]; then
        log "Running tests..."
        
        # Run tests in Docker container
        docker-compose -f docker-compose.yml --profile testing run --rm test
        
        success "Tests completed successfully"
    else
        warn "Skipping tests"
    fi
}

# Build Docker image
build_docker_image() {
    local target=$1
    local tag="hf-eco2ai-plugin:${target}"
    
    log "Building Docker image: $tag"
    
    docker build \
        --target "$target" \
        --tag "$tag" \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg VCS_REF="$VCS_REF" \
        --build-arg VERSION="$VERSION" \
        --file Dockerfile \
        .
    
    success "Docker image built: $tag"
}

# Tag images for registry
tag_for_registry() {
    local source_tag=$1
    local registry_prefix="${DOCKER_REGISTRY:-ghcr.io/terragonlabs}"
    
    # Tag with version
    docker tag "$source_tag" "${registry_prefix}/hf-eco2ai-plugin:${VERSION}"
    
    # Tag with latest if on main branch
    if [[ "$BRANCH" == "main" ]]; then
        docker tag "$source_tag" "${registry_prefix}/hf-eco2ai-plugin:latest"
    fi
    
    # Tag with branch name
    docker tag "$source_tag" "${registry_prefix}/hf-eco2ai-plugin:${BRANCH}"
    
    success "Tagged image for registry: $source_tag"
}

# Push images to registry
push_images() {
    if [[ "$PUSH_IMAGES" == "true" ]]; then
        log "Pushing images to registry..."
        
        local registry_prefix="${DOCKER_REGISTRY:-ghcr.io/terragonlabs}"
        
        # Push version tag
        docker push "${registry_prefix}/hf-eco2ai-plugin:${VERSION}"
        
        # Push latest if on main branch
        if [[ "$BRANCH" == "main" ]]; then
            docker push "${registry_prefix}/hf-eco2ai-plugin:latest"
        fi
        
        # Push branch tag
        docker push "${registry_prefix}/hf-eco2ai-plugin:${BRANCH}"
        
        success "Images pushed to registry"
    else
        warn "Skipping image push"
    fi
}

# Generate build report
generate_build_report() {
    local report_file="build-report.json"
    
    log "Generating build report..."
    
    cat > "$report_file" << EOF
{
  "build_info": {
    "timestamp": "$BUILD_DATE",
    "version": "$VERSION",
    "vcs_ref": "$VCS_REF",
    "branch": "$BRANCH",
    "build_type": "$BUILD_TYPE"
  },
  "environment": {
    "docker_version": "$(docker --version | cut -d' ' -f3 | tr -d ',')",
    "docker_compose_version": "$(docker-compose --version | cut -d' ' -f3 | tr -d ',')",
    "python_version": "$(python --version | cut -d' ' -f2)",
    "platform": "$(uname -s -m)"
  },
  "artifacts": {
    "python_package": true,
    "docker_images": [
$(docker images --format '      "{{.Repository}}:{{.Tag}}",' | grep hf-eco2ai-plugin | sed '$s/,$//')
    ]
  }
}
EOF
    
    success "Build report generated: $report_file"
}

# Main build function
main() {
    log "Starting HF Eco2AI Plugin build..."
    log "Build type: $BUILD_TYPE"
    log "Version: $VERSION"
    log "VCS Ref: $VCS_REF"
    log "Branch: $BRANCH"
    
    check_prerequisites
    clean_build_artifacts
    
    # Build Python package
    build_python_package
    
    # Run tests
    run_tests
    
    # Build Docker images
    case "$BUILD_TYPE" in
        "production")
            build_docker_image "production"
            tag_for_registry "hf-eco2ai-plugin:production"
            ;;
        "development")
            build_docker_image "development"
            tag_for_registry "hf-eco2ai-plugin:development"
            ;;
        "all")
            build_docker_image "production"
            build_docker_image "development"
            tag_for_registry "hf-eco2ai-plugin:production"
            tag_for_registry "hf-eco2ai-plugin:development"
            ;;
    esac
    
    # Push images if requested
    push_images
    
    # Generate build report
    generate_build_report
    
    success "Build completed successfully!"
}

# Error handling
trap 'error "Build failed at line $LINENO"' ERR

# Run main function
main "$@"
