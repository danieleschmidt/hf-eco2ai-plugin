#!/bin/bash
# Security scanning script for HF Eco2AI Plugin

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCAN_DOCKER_IMAGES=true
SCAN_DEPENDENCIES=true
SCAN_CODE=true
SCAN_SECRETS=true
FAIL_ON_HIGH=true
FAIL_ON_CRITICAL=true
OUTPUT_FORMAT="json"
REPORT_DIR="security-reports"

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run security scans for HF Eco2AI Plugin

OPTIONS:
    --no-docker         Skip Docker image security scanning
    --no-deps           Skip dependency vulnerability scanning
    --no-code           Skip static code analysis
    --no-secrets        Skip secret detection
    --no-fail-high      Don't fail on HIGH severity issues
    --no-fail-critical  Don't fail on CRITICAL severity issues
    --format FORMAT     Output format: json, table, sarif (default: json)
    --output DIR        Output directory for reports (default: security-reports)
    -h, --help          Show this help message

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
        --no-docker)
            SCAN_DOCKER_IMAGES=false
            shift
            ;;
        --no-deps)
            SCAN_DEPENDENCIES=false
            shift
            ;;
        --no-code)
            SCAN_CODE=false
            shift
            ;;
        --no-secrets)
            SCAN_SECRETS=false
            shift
            ;;
        --no-fail-high)
            FAIL_ON_HIGH=false
            shift
            ;;
        --no-fail-critical)
            FAIL_ON_CRITICAL=false
            shift
            ;;
        --format)
            OUTPUT_FORMAT="$2"
            shift 2
            ;;
        --output)
            REPORT_DIR="$2"
            shift 2
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

# Create output directory
mkdir -p "$REPORT_DIR"

# Check if required tools are available
check_tools() {
    log "Checking security tools..."
    
    local missing_tools=()
    
    # Check for dependency scanning tools
    if [[ "$SCAN_DEPENDENCIES" == "true" ]]; then
        if ! command -v safety &> /dev/null; then
            missing_tools+=("safety")
        fi
        if ! command -v pip-audit &> /dev/null; then
            missing_tools+=("pip-audit")
        fi
    fi
    
    # Check for code analysis tools
    if [[ "$SCAN_CODE" == "true" ]]; then
        if ! command -v bandit &> /dev/null; then
            missing_tools+=("bandit")
        fi
    fi
    
    # Check for secret detection tools
    if [[ "$SCAN_SECRETS" == "true" ]]; then
        if ! command -v detect-secrets &> /dev/null; then
            missing_tools+=("detect-secrets")
        fi
    fi
    
    # Check for Docker scanning tools
    if [[ "$SCAN_DOCKER_IMAGES" == "true" ]]; then
        if ! command -v trivy &> /dev/null && ! command -v docker &> /dev/null; then
            warn "Neither trivy nor docker scan available for image scanning"
        fi
    fi
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        error "Missing required tools: ${missing_tools[*]}"
        log "Install missing tools with: pip install ${missing_tools[*]}"
        exit 1
    fi
    
    success "Security tools check passed"
}

# Scan Python dependencies for vulnerabilities
scan_dependencies() {
    if [[ "$SCAN_DEPENDENCIES" != "true" ]]; then
        return 0
    fi
    
    log "Scanning Python dependencies for vulnerabilities..."
    
    local exit_code=0
    
    # Safety scan
    if command -v safety &> /dev/null; then
        log "Running Safety scan..."
        if ! safety check --json --output "$REPORT_DIR/safety-report.json"; then
            warn "Safety scan found vulnerabilities"
            exit_code=1
        fi
    fi
    
    # pip-audit scan
    if command -v pip-audit &> /dev/null; then
        log "Running pip-audit scan..."
        if ! pip-audit --format="$OUTPUT_FORMAT" --output="$REPORT_DIR/pip-audit-report.$OUTPUT_FORMAT"; then
            warn "pip-audit found vulnerabilities"
            exit_code=1
        fi
    fi
    
    if [[ $exit_code -eq 0 ]]; then
        success "Dependency scan completed - no vulnerabilities found"
    else
        warn "Dependency scan completed - vulnerabilities found"
    fi
    
    return $exit_code
}

# Scan code for security issues
scan_code() {
    if [[ "$SCAN_CODE" != "true" ]]; then
        return 0
    fi
    
    log "Running static code analysis..."
    
    local exit_code=0
    
    # Bandit scan
    if command -v bandit &> /dev/null; then
        log "Running Bandit security scan..."
        if ! bandit -r src/ -f "$OUTPUT_FORMAT" -o "$REPORT_DIR/bandit-report.$OUTPUT_FORMAT"; then
            warn "Bandit found security issues"
            exit_code=1
        fi
    fi
    
    if [[ $exit_code -eq 0 ]]; then
        success "Code security scan completed - no issues found"
    else
        warn "Code security scan completed - issues found"
    fi
    
    return $exit_code
}

# Scan for secrets in code
scan_secrets() {
    if [[ "$SCAN_SECRETS" != "true" ]]; then
        return 0
    fi
    
    log "Scanning for secrets in code..."
    
    local exit_code=0
    
    if command -v detect-secrets &> /dev/null; then
        log "Running detect-secrets scan..."
        
        # Create baseline if it doesn't exist
        if [[ ! -f .secrets.baseline ]]; then
            detect-secrets scan --baseline .secrets.baseline
        fi
        
        # Scan for new secrets
        if ! detect-secrets scan --baseline .secrets.baseline --exclude-files="$REPORT_DIR/.*"; then
            warn "New secrets detected"
            exit_code=1
        fi
        
        # Generate report
        detect-secrets audit .secrets.baseline --report --output "$REPORT_DIR/secrets-report.json"
    fi
    
    if [[ $exit_code -eq 0 ]]; then
        success "Secret scan completed - no new secrets found"
    else
        warn "Secret scan completed - new secrets found"
    fi
    
    return $exit_code
}

# Scan Docker images for vulnerabilities
scan_docker_images() {
    if [[ "$SCAN_DOCKER_IMAGES" != "true" ]]; then
        return 0
    fi
    
    log "Scanning Docker images for vulnerabilities..."
    
    local exit_code=0
    local images=("hf-eco2ai-plugin:production" "hf-eco2ai-plugin:development")
    
    for image in "${images[@]}"; do
        if docker image inspect "$image" &> /dev/null; then
            log "Scanning image: $image"
            
            # Try Trivy first (more comprehensive)
            if command -v trivy &> /dev/null; then
                if ! trivy image --format "$OUTPUT_FORMAT" --output "$REPORT_DIR/trivy-$(echo "$image" | tr ':' '-').${OUTPUT_FORMAT}" "$image"; then
                    warn "Trivy found vulnerabilities in $image"
                    exit_code=1
                fi
            # Fallback to docker scan
            elif command -v docker &> /dev/null && docker scan --help &> /dev/null; then
                if ! docker scan --json "$image" > "$REPORT_DIR/docker-scan-$(echo "$image" | tr ':' '-').json"; then
                    warn "Docker scan found vulnerabilities in $image"
                    exit_code=1
                fi
            else
                warn "No Docker vulnerability scanner available"
            fi
        else
            warn "Image not found: $image"
        fi
    done
    
    if [[ $exit_code -eq 0 ]]; then
        success "Docker image scan completed - no vulnerabilities found"
    else
        warn "Docker image scan completed - vulnerabilities found"
    fi
    
    return $exit_code
}

# Generate security summary report
generate_summary() {
    log "Generating security summary..."
    
    local summary_file="$REPORT_DIR/security-summary.json"
    local timestamp=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
    
    cat > "$summary_file" << EOF
{
  "scan_info": {
    "timestamp": "$timestamp",
    "scans_performed": {
      "dependencies": $SCAN_DEPENDENCIES,
      "code_analysis": $SCAN_CODE,
      "secret_detection": $SCAN_SECRETS,
      "docker_images": $SCAN_DOCKER_IMAGES
    }
  },
  "report_files": [
EOF
    
    # List all generated report files
    find "$REPORT_DIR" -name "*.json" -o -name "*.sarif" -o -name "*.txt" | while read -r file; do
        echo "    \"$(basename "$file")\",$" >> "$summary_file"
    done
    
    # Remove trailing comma and close JSON
    sed -i '$ s/,$//' "$summary_file"
    cat >> "$summary_file" << EOF
  ],
  "scan_status": "completed"
}
EOF
    
    success "Security summary generated: $summary_file"
}

# Main function
main() {
    log "Starting security scan for HF Eco2AI Plugin..."
    
    check_tools
    
    local overall_exit_code=0
    
    # Run all scans
    if ! scan_dependencies; then
        overall_exit_code=1
    fi
    
    if ! scan_code; then
        overall_exit_code=1
    fi
    
    if ! scan_secrets; then
        overall_exit_code=1
    fi
    
    if ! scan_docker_images; then
        overall_exit_code=1
    fi
    
    # Generate summary
    generate_summary
    
    # Final result
    if [[ $overall_exit_code -eq 0 ]]; then
        success "All security scans passed!"
    else
        if [[ "$FAIL_ON_HIGH" == "true" ]] || [[ "$FAIL_ON_CRITICAL" == "true" ]]; then
            error "Security scans failed - vulnerabilities found"
            exit 1
        else
            warn "Security scans completed with warnings - vulnerabilities found but not failing"
        fi
    fi
    
    log "Security reports available in: $REPORT_DIR"
}

# Error handling
trap 'error "Security scan failed at line $LINENO"' ERR

# Run main function
main "$@"
