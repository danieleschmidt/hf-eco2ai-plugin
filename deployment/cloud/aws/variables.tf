# Variables for AWS EKS deployment

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  default     = "hf-eco2ai-cluster"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name (development, staging, production)"
  type        = string
  
  validation {
    condition     = contains(["development", "staging", "production"], var.environment)
    error_message = "Environment must be one of: development, staging, production."
  }
}

variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

# RDS Configuration
variable "rds_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
  
  validation {
    condition     = can(regex("^db\\.", var.rds_instance_class))
    error_message = "RDS instance class must start with 'db.'."
  }
}

variable "rds_allocated_storage" {
  description = "RDS allocated storage in GB"
  type        = number
  default     = 20
  
  validation {
    condition     = var.rds_allocated_storage >= 20 && var.rds_allocated_storage <= 65536
    error_message = "RDS allocated storage must be between 20 and 65536 GB."
  }
}

variable "rds_max_allocated_storage" {
  description = "RDS maximum allocated storage in GB"
  type        = number
  default     = 100
}

variable "rds_password" {
  description = "RDS master password"
  type        = string
  sensitive   = true
  
  validation {
    condition     = length(var.rds_password) >= 8
    error_message = "RDS password must be at least 8 characters long."
  }
}

# Redis Configuration
variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.t3.micro"
  
  validation {
    condition     = can(regex("^cache\\.", var.redis_node_type))
    error_message = "Redis node type must start with 'cache.'."
  }
}

variable "redis_num_cache_nodes" {
  description = "Number of cache nodes"
  type        = number
  default     = 1
  
  validation {
    condition     = var.redis_num_cache_nodes >= 1 && var.redis_num_cache_nodes <= 6
    error_message = "Number of cache nodes must be between 1 and 6."
  }
}

variable "redis_auth_token" {
  description = "Redis AUTH token"
  type        = string
  sensitive   = true
  default     = null
}

# S3 Configuration
variable "s3_bucket_name" {
  description = "S3 bucket name for data storage"
  type        = string
  
  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9-]*[a-z0-9]$", var.s3_bucket_name))
    error_message = "S3 bucket name must be lowercase alphanumeric with hyphens."
  }
}

# Networking
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
  
  validation {
    condition     = can(cidrhost(var.vpc_cidr, 0))
    error_message = "VPC CIDR must be a valid IPv4 CIDR block."
  }
}

variable "availability_zones" {
  description = "Availability zones"
  type        = list(string)
  default     = []
}

# Node Groups Configuration
variable "system_node_group" {
  description = "System node group configuration"
  type = object({
    instance_types = list(string)
    min_size      = number
    max_size      = number
    desired_size  = number
  })
  default = {
    instance_types = ["t3.medium"]
    min_size      = 1
    max_size      = 3
    desired_size  = 2
  }
}

variable "application_node_group" {
  description = "Application node group configuration"
  type = object({
    instance_types = list(string)
    min_size      = number
    max_size      = number
    desired_size  = number
    disk_size     = number
  })
  default = {
    instance_types = ["t3.large", "t3.xlarge"]
    min_size      = 2
    max_size      = 10
    desired_size  = 3
    disk_size     = 50
  }
}

variable "ml_node_group" {
  description = "ML workloads node group configuration"
  type = object({
    instance_types = list(string)
    min_size      = number
    max_size      = number
    desired_size  = number
    disk_size     = number
  })
  default = {
    instance_types = ["c5.2xlarge", "c5.4xlarge"]
    min_size      = 0
    max_size      = 5
    desired_size  = 1
    disk_size     = 100
  }
}

# Monitoring and Logging
variable "enable_cloudwatch_logs" {
  description = "Enable CloudWatch logs"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 7
  
  validation {
    condition     = contains([1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653], var.log_retention_days)
    error_message = "Log retention days must be a valid CloudWatch retention period."
  }
}

# Security
variable "enable_irsa" {
  description = "Enable IAM Roles for Service Accounts"
  type        = bool
  default     = true
}

variable "enable_encryption" {
  description = "Enable encryption at rest"
  type        = bool
  default     = true
}

variable "enable_network_policy" {
  description = "Enable Kubernetes network policies"
  type        = bool
  default     = true
}

# Backup and Recovery
variable "backup_retention_period" {
  description = "Backup retention period in days"
  type        = number
  default     = 7
  
  validation {
    condition     = var.backup_retention_period >= 0 && var.backup_retention_period <= 35
    error_message = "Backup retention period must be between 0 and 35 days."
  }
}

variable "enable_point_in_time_recovery" {
  description = "Enable point in time recovery"
  type        = bool
  default     = true
}

# Cost Optimization
variable "enable_spot_instances" {
  description = "Enable spot instances for cost optimization"
  type        = bool
  default     = false
}

variable "spot_instance_percentage" {
  description = "Percentage of spot instances"
  type        = number
  default     = 50
  
  validation {
    condition     = var.spot_instance_percentage >= 0 && var.spot_instance_percentage <= 100
    error_message = "Spot instance percentage must be between 0 and 100."
  }
}

# Tags
variable "tags" {
  description = "A map of tags to add to all resources"
  type        = map(string)
  default = {
    Project     = "hf-eco2ai"
    ManagedBy   = "terraform"
    Environment = "development"
  }
}

# Advanced Configuration
variable "enable_cluster_autoscaler" {
  description = "Enable cluster autoscaler"
  type        = bool
  default     = true
}

variable "enable_metrics_server" {
  description = "Enable metrics server"
  type        = bool
  default     = true
}

variable "enable_aws_load_balancer_controller" {
  description = "Enable AWS Load Balancer Controller"
  type        = bool
  default     = true
}

variable "enable_external_dns" {
  description = "Enable ExternalDNS"
  type        = bool
  default     = false
}

variable "external_dns_domain" {
  description = "Domain for ExternalDNS"
  type        = string
  default     = ""
}

# Application Configuration
variable "hf_eco2ai_image_tag" {
  description = "HF Eco2AI Docker image tag"
  type        = string
  default     = "latest"
}

variable "hf_eco2ai_replicas" {
  description = "Number of HF Eco2AI replicas"
  type        = number
  default     = 3
  
  validation {
    condition     = var.hf_eco2ai_replicas >= 1 && var.hf_eco2ai_replicas <= 100
    error_message = "Number of replicas must be between 1 and 100."
  }
}

variable "hf_eco2ai_resources" {
  description = "Resource requests and limits for HF Eco2AI"
  type = object({
    requests = object({
      cpu    = string
      memory = string
    })
    limits = object({
      cpu    = string
      memory = string
    })
  })
  default = {
    requests = {
      cpu    = "500m"
      memory = "1Gi"
    }
    limits = {
      cpu    = "2"
      memory = "4Gi"
    }
  }
}

# Autoscaling Configuration
variable "hpa_config" {
  description = "Horizontal Pod Autoscaler configuration"
  type = object({
    min_replicas                     = number
    max_replicas                     = number
    target_cpu_utilization_percentage = number
    target_memory_utilization_percentage = number
  })
  default = {
    min_replicas                     = 3
    max_replicas                     = 10
    target_cpu_utilization_percentage = 70
    target_memory_utilization_percentage = 80
  }
}

# Persistence Configuration
variable "persistence_config" {
  description = "Persistence configuration"
  type = object({
    enabled      = bool
    storage_class = string
    size         = string
  })
  default = {
    enabled      = true
    storage_class = "gp3"
    size         = "10Gi"
  }
}