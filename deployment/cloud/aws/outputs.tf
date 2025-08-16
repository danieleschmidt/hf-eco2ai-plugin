# Outputs for AWS EKS deployment

# Cluster Information
output "cluster_name" {
  description = "Name of the EKS cluster"
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "cluster_iam_role_name" {
  description = "IAM role name associated with EKS cluster"
  value       = module.eks.cluster_iam_role_name
}

output "cluster_iam_role_arn" {
  description = "IAM role ARN associated with EKS cluster"
  value       = module.eks.cluster_iam_role_arn
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
}

output "cluster_primary_security_group_id" {
  description = "Cluster security group that was created by Amazon EKS for the cluster"
  value       = module.eks.cluster_primary_security_group_id
}

output "cluster_service_cidr" {
  description = "The CIDR block where Kubernetes pod and service IP addresses are assigned from"
  value       = module.eks.cluster_service_cidr
}

output "cluster_ip_family" {
  description = "The IP family used to assign Kubernetes pod and service addresses"
  value       = module.eks.cluster_ip_family
}

# OIDC Provider
output "oidc_provider_arn" {
  description = "The ARN of the OIDC Provider if enabled"
  value       = module.eks.oidc_provider_arn
}

output "cluster_oidc_issuer_url" {
  description = "The URL on the EKS cluster for the OpenID Connect identity provider"
  value       = module.eks.cluster_oidc_issuer_url
}

# Node Groups
output "eks_managed_node_groups" {
  description = "Map of attribute maps for all EKS managed node groups created"
  value       = module.eks.eks_managed_node_groups
}

output "eks_managed_node_groups_autoscaling_group_names" {
  description = "List of the autoscaling group names created by EKS managed node groups"
  value       = module.eks.eks_managed_node_groups_autoscaling_group_names
}

# VPC Information
output "vpc_id" {
  description = "ID of the VPC where the cluster is deployed"
  value       = module.vpc.vpc_id
}

output "vpc_cidr_block" {
  description = "The CIDR block of the VPC"
  value       = module.vpc.vpc_cidr_block
}

output "private_subnets" {
  description = "List of IDs of private subnets"
  value       = module.vpc.private_subnets
}

output "public_subnets" {
  description = "List of IDs of public subnets"
  value       = module.vpc.public_subnets
}

output "intra_subnets" {
  description = "List of IDs of intra subnets"
  value       = module.vpc.intra_subnets
}

output "nat_gateway_ids" {
  description = "List of IDs of the NAT Gateways"
  value       = module.vpc.natgw_ids
}

output "internet_gateway_id" {
  description = "The ID of the Internet Gateway"
  value       = module.vpc.igw_id
}

# RDS Information
output "rds_endpoint" {
  description = "RDS instance endpoint"
  value       = aws_db_instance.hf_eco2ai.endpoint
  sensitive   = true
}

output "rds_port" {
  description = "RDS instance port"
  value       = aws_db_instance.hf_eco2ai.port
}

output "rds_db_name" {
  description = "RDS database name"
  value       = aws_db_instance.hf_eco2ai.db_name
}

output "rds_username" {
  description = "RDS master username"
  value       = aws_db_instance.hf_eco2ai.username
  sensitive   = true
}

output "rds_security_group_id" {
  description = "RDS security group ID"
  value       = aws_security_group.rds.id
}

# Redis Information
output "redis_endpoint" {
  description = "Redis primary endpoint"
  value       = aws_elasticache_replication_group.hf_eco2ai.primary_endpoint_address
  sensitive   = true
}

output "redis_port" {
  description = "Redis port"
  value       = aws_elasticache_replication_group.hf_eco2ai.port
}

output "redis_auth_token_enabled" {
  description = "Whether Redis AUTH token is enabled"
  value       = aws_elasticache_replication_group.hf_eco2ai.auth_token != null
}

# S3 Information
output "s3_bucket_name" {
  description = "Name of the S3 bucket"
  value       = aws_s3_bucket.hf_eco2ai_data.bucket
}

output "s3_bucket_arn" {
  description = "ARN of the S3 bucket"
  value       = aws_s3_bucket.hf_eco2ai_data.arn
}

output "s3_bucket_domain_name" {
  description = "Domain name of the S3 bucket"
  value       = aws_s3_bucket.hf_eco2ai_data.bucket_domain_name
}

# Load Balancer Information
output "load_balancer_dns_name" {
  description = "DNS name of the load balancer"
  value       = aws_lb.hf_eco2ai.dns_name
}

output "load_balancer_zone_id" {
  description = "Zone ID of the load balancer"
  value       = aws_lb.hf_eco2ai.zone_id
}

output "load_balancer_arn" {
  description = "ARN of the load balancer"
  value       = aws_lb.hf_eco2ai.arn
}

# IAM Roles
output "hf_eco2ai_iam_role_arn" {
  description = "ARN of HF Eco2AI IAM role"
  value       = aws_iam_role.hf_eco2ai_role.arn
}

output "load_balancer_controller_iam_role_arn" {
  description = "ARN of AWS Load Balancer Controller IAM role"
  value       = module.load_balancer_controller_irsa_role.iam_role_arn
}

output "ebs_csi_iam_role_arn" {
  description = "ARN of EBS CSI driver IAM role"
  value       = module.ebs_csi_irsa_role.iam_role_arn
}

# Security Groups
output "additional_security_group_id" {
  description = "ID of the additional security group"
  value       = aws_security_group.additional.id
}

output "alb_security_group_id" {
  description = "ID of the ALB security group"
  value       = aws_security_group.alb.id
}

# KMS
output "kms_key_id" {
  description = "The globally unique identifier for the key"
  value       = aws_kms_key.eks.key_id
}

output "kms_key_arn" {
  description = "The Amazon Resource Name (ARN) of the key"
  value       = aws_kms_key.eks.arn
}

# Connection Information
output "kubectl_config" {
  description = "kubectl config as generated by the module"
  value = templatefile("${path.module}/kubeconfig.tpl", {
    cluster_name           = module.eks.cluster_name
    endpoint              = module.eks.cluster_endpoint
    ca_certificate        = module.eks.cluster_certificate_authority_data
    aws_region            = var.aws_region
  })
  sensitive = true
}

# Application URLs
output "application_urls" {
  description = "Application access URLs"
  value = {
    load_balancer = "https://${aws_lb.hf_eco2ai.dns_name}"
    health_check  = "https://${aws_lb.hf_eco2ai.dns_name}/health"
    metrics       = "https://${aws_lb.hf_eco2ai.dns_name}/metrics"
  }
}

# Environment Information
output "environment_info" {
  description = "Environment configuration summary"
  value = {
    cluster_name    = local.cluster_name
    environment     = var.environment
    region          = var.aws_region
    kubernetes_version = var.kubernetes_version
    vpc_cidr        = local.vpc_cidr
    availability_zones = local.azs
  }
}

# Cost Information
output "estimated_monthly_cost" {
  description = "Estimated monthly cost breakdown (USD)"
  value = {
    note = "Estimates based on us-west-2 pricing, actual costs may vary"
    eks_cluster = "73.00"  # $0.10/hour
    nodes_compute = "varies based on instance types and count"
    rds = "varies based on instance class and storage"
    redis = "varies based on node type and count"
    data_transfer = "varies based on usage"
    storage = "varies based on EBS and S3 usage"
    load_balancer = "16.43"  # $0.0225/hour for ALB
  }
}

# Monitoring Endpoints
output "monitoring_endpoints" {
  description = "Monitoring and observability endpoints"
  value = {
    prometheus = "http://prometheus.monitoring.svc.cluster.local:9090"
    grafana    = "http://grafana.monitoring.svc.cluster.local:3000"
    jaeger     = "http://jaeger.monitoring.svc.cluster.local:16686"
    alertmanager = "http://alertmanager.monitoring.svc.cluster.local:9093"
  }
}

# Backup Information
output "backup_configuration" {
  description = "Backup configuration details"
  value = {
    s3_bucket = aws_s3_bucket.hf_eco2ai_data.bucket
    rds_backup_retention = aws_db_instance.hf_eco2ai.backup_retention_period
    redis_snapshot_retention = aws_elasticache_replication_group.hf_eco2ai.snapshot_retention_limit
  }
}