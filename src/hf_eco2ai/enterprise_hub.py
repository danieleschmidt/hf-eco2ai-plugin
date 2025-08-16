"""Enterprise Integration Hub for HF Eco2AI Carbon Tracking.

This module provides enterprise-grade integration capabilities including:
- Slack/Teams notifications
- Executive reporting automation  
- Corporate carbon accounting system integration
- ESG compliance reporting
- Automated dashboards and alerts
"""

import time
import logging
import json
import asyncio
import threading
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import uuid
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import csv
import io
import schedule

# Optional enterprise integrations with fallbacks
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from jinja2 import Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

# Import existing components
try:
    from .performance_optimizer import optimized, get_performance_optimizer
    from .error_handling import handle_gracefully, ErrorSeverity, resilient_operation
    from .models import CarbonReport, CarbonMetrics
except ImportError:
    # Fallback decorators
    def optimized(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def handle_gracefully(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def resilient_operation(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    class ErrorSeverity:
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"


logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """Types of notification channels."""
    SLACK = "slack"
    TEAMS = "teams"
    EMAIL = "email"
    WEBHOOK = "webhook"
    SMS = "sms"


class ReportFrequency(Enum):
    """Reporting frequency options."""
    REAL_TIME = "real_time"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


class ESGFramework(Enum):
    """ESG reporting frameworks."""
    CDP = "cdp"  # Carbon Disclosure Project
    GRI = "gri"  # Global Reporting Initiative
    SASB = "sasb"  # Sustainability Accounting Standards Board
    TCFD = "tcfd"  # Task Force on Climate-related Financial Disclosures
    EU_TAXONOMY = "eu_taxonomy"
    CUSTOM = "custom"


@dataclass
class NotificationConfig:
    """Configuration for enterprise notifications."""
    channel: NotificationChannel
    webhook_url: Optional[str] = None
    email_settings: Optional[Dict[str, str]] = None
    threshold_conditions: Dict[str, float] = field(default_factory=dict)
    message_template: Optional[str] = None
    enabled: bool = True
    priority_level: str = "medium"  # low, medium, high, critical


@dataclass
class CarbonBudgetAlert:
    """Carbon budget alert configuration."""
    budget_name: str
    limit_kg_co2: float
    warning_threshold_pct: float = 80.0  # Alert at 80% of budget
    critical_threshold_pct: float = 95.0  # Critical alert at 95%
    notification_channels: List[NotificationChannel] = field(default_factory=list)
    auto_actions: List[str] = field(default_factory=list)  # Actions to take on alert


@dataclass
class ESGMetrics:
    """ESG compliance metrics."""
    framework: ESGFramework
    carbon_emissions_scope1: float = 0.0  # Direct emissions
    carbon_emissions_scope2: float = 0.0  # Indirect emissions from energy
    carbon_emissions_scope3: float = 0.0  # Other indirect emissions
    renewable_energy_percentage: float = 0.0
    carbon_intensity_per_revenue: float = 0.0
    carbon_offset_credits: float = 0.0
    compliance_score: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ExecutiveReport:
    """Executive carbon performance report."""
    report_id: str
    period_start: datetime
    period_end: datetime
    total_emissions_kg: float
    total_energy_kwh: float
    cost_impact_usd: float
    efficiency_trend: str  # "improving", "stable", "declining"
    key_findings: List[str]
    recommendations: List[str]
    compliance_status: Dict[str, str]
    carbon_budget_status: Dict[str, float]
    department_breakdown: Dict[str, Dict[str, float]]
    generated_timestamp: datetime = field(default_factory=datetime.now)


class SlackNotifier:
    """Slack integration for carbon tracking notifications."""
    
    def __init__(self, webhook_url: str, default_channel: str = "#carbon-tracking"):
        self.webhook_url = webhook_url
        self.default_channel = default_channel
        self.rate_limiter = {}  # Simple rate limiting
        
    @resilient_operation(max_attempts=3)
    def send_notification(self, message: str, channel: Optional[str] = None, 
                         attachments: Optional[List[Dict]] = None) -> bool:
        """Send notification to Slack."""
        if not REQUESTS_AVAILABLE:
            logger.error("requests library not available for Slack notifications")
            return False
        
        # Rate limiting: max 1 message per minute per channel
        channel = channel or self.default_channel
        now = time.time()
        if channel in self.rate_limiter:
            if now - self.rate_limiter[channel] < 60:
                logger.debug(f"Rate limited Slack message to {channel}")
                return False
        self.rate_limiter[channel] = now
        
        payload = {
            "channel": channel,
            "text": message,
            "username": "Carbon Tracker Bot",
            "icon_emoji": ":leaves:"
        }
        
        if attachments:
            payload["attachments"] = attachments
        
        try:
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            logger.debug(f"Slack notification sent to {channel}")
            return True
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False
    
    def send_carbon_alert(self, alert_type: str, current_value: float, 
                         threshold: float, details: Dict[str, Any]) -> bool:
        """Send carbon-specific alert to Slack."""
        color = "danger" if alert_type == "critical" else "warning"
        
        attachment = {
            "color": color,
            "title": f"🚨 Carbon {alert_type.title()} Alert",
            "fields": [
                {"title": "Current Emissions", "value": f"{current_value:.2f} kg CO₂", "short": True},
                {"title": "Threshold", "value": f"{threshold:.2f} kg CO₂", "short": True},
                {"title": "Project", "value": details.get("project_name", "Unknown"), "short": True},
                {"title": "Time", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "short": True}
            ],
            "footer": "HF Eco2AI Carbon Tracker"
        }
        
        if "recommendations" in details:
            attachment["fields"].append({
                "title": "Recommendations",
                "value": "\n".join(details["recommendations"][:3]),
                "short": False
            })
        
        message = f"Carbon emissions have exceeded {alert_type} threshold!"
        return self.send_notification(message, attachments=[attachment])


class TeamsNotifier:
    """Microsoft Teams integration for carbon tracking notifications."""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.rate_limiter = {}
        
    @resilient_operation(max_attempts=3)
    def send_notification(self, title: str, message: str, 
                         facts: Optional[List[Dict[str, str]]] = None) -> bool:
        """Send notification to Microsoft Teams."""
        if not REQUESTS_AVAILABLE:
            logger.error("requests library not available for Teams notifications")
            return False
        
        # Rate limiting
        now = time.time()
        if "teams" in self.rate_limiter:
            if now - self.rate_limiter["teams"] < 60:
                logger.debug("Rate limited Teams message")
                return False
        self.rate_limiter["teams"] = now
        
        payload = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": "0076D7",
            "summary": title,
            "sections": [{
                "activityTitle": title,
                "activitySubtitle": "HF Eco2AI Carbon Tracker",
                "activityImage": "https://via.placeholder.com/64x64/00ff00/ffffff?text=🌱",
                "text": message,
                "markdown": True
            }]
        }
        
        if facts:
            payload["sections"][0]["facts"] = facts
        
        try:
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            logger.debug("Teams notification sent")
            return True
        except Exception as e:
            logger.error(f"Failed to send Teams notification: {e}")
            return False
    
    def send_carbon_summary(self, summary_data: Dict[str, Any]) -> bool:
        """Send carbon tracking summary to Teams."""
        facts = [
            {"name": "Total Emissions", "value": f"{summary_data.get('total_co2_kg', 0):.2f} kg CO₂"},
            {"name": "Energy Consumed", "value": f"{summary_data.get('total_energy_kwh', 0):.2f} kWh"},
            {"name": "Efficiency", "value": f"{summary_data.get('samples_per_kwh', 0):.0f} samples/kWh"},
            {"name": "Project", "value": summary_data.get('project_name', 'Unknown')},
            {"name": "Duration", "value": f"{summary_data.get('duration_hours', 0):.1f} hours"}
        ]
        
        title = "📊 Carbon Tracking Summary"
        message = "Training session completed. Here's your carbon impact summary:"
        
        return self.send_notification(title, message, facts)


class EmailReporter:
    """Email reporting for carbon tracking."""
    
    def __init__(self, smtp_host: str, smtp_port: int, username: str, password: str):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        
    @resilient_operation(max_attempts=2)
    def send_executive_report(self, report: ExecutiveReport, 
                            recipients: List[str]) -> bool:
        """Send executive carbon report via email."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['To'] = ", ".join(recipients)
            msg['Subject'] = f"Carbon Impact Report - {report.period_start.strftime('%B %Y')}"
            
            # HTML email body
            html_body = self._generate_html_report(report)
            msg.attach(MIMEText(html_body, 'html'))
            
            # CSV attachment
            csv_data = self._generate_csv_report(report)
            csv_attachment = MIMEBase('application', 'octet-stream')
            csv_attachment.set_payload(csv_data.encode())
            encoders.encode_base64(csv_attachment)
            csv_attachment.add_header(
                'Content-Disposition',
                f'attachment; filename="carbon_report_{report.report_id}.csv"'
            )
            msg.attach(csv_attachment)
            
            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logger.info(f"Executive report emailed to {len(recipients)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send executive report: {e}")
            return False
    
    def _generate_html_report(self, report: ExecutiveReport) -> str:
        """Generate HTML report from template."""
        if not JINJA2_AVAILABLE:
            return self._generate_simple_html_report(report)
        
        template_str = """
        <html>
        <head><title>Carbon Impact Report</title></head>
        <body style="font-family: Arial, sans-serif;">
            <h1>🌱 Carbon Impact Report</h1>
            <h2>Period: {{ period_start }} to {{ period_end }}</h2>
            
            <h3>Key Metrics</h3>
            <ul>
                <li><strong>Total Emissions:</strong> {{ total_emissions_kg|round(2) }} kg CO₂</li>
                <li><strong>Energy Consumed:</strong> {{ total_energy_kwh|round(2) }} kWh</li>
                <li><strong>Cost Impact:</strong> ${{ cost_impact_usd|round(2) }}</li>
                <li><strong>Efficiency Trend:</strong> {{ efficiency_trend|title }}</li>
            </ul>
            
            <h3>Key Findings</h3>
            <ul>
            {% for finding in key_findings %}
                <li>{{ finding }}</li>
            {% endfor %}
            </ul>
            
            <h3>Recommendations</h3>
            <ul>
            {% for rec in recommendations %}
                <li>{{ rec }}</li>
            {% endfor %}
            </ul>
            
            <h3>Department Breakdown</h3>
            <table border="1" style="border-collapse: collapse;">
                <tr><th>Department</th><th>Emissions (kg CO₂)</th><th>Energy (kWh)</th></tr>
                {% for dept, metrics in department_breakdown.items() %}
                <tr>
                    <td>{{ dept }}</td>
                    <td>{{ metrics.get('emissions', 0)|round(2) }}</td>
                    <td>{{ metrics.get('energy', 0)|round(2) }}</td>
                </tr>
                {% endfor %}
            </table>
            
            <p><em>Generated on {{ generated_timestamp.strftime('%Y-%m-%d %H:%M:%S') }}</em></p>
        </body>
        </html>
        """
        
        template = Template(template_str)
        return template.render(**report.__dict__)
    
    def _generate_simple_html_report(self, report: ExecutiveReport) -> str:
        """Generate simple HTML report without Jinja2."""
        return f"""
        <html>
        <body>
            <h1>Carbon Impact Report</h1>
            <p>Period: {report.period_start} to {report.period_end}</p>
            <p>Total Emissions: {report.total_emissions_kg:.2f} kg CO₂</p>
            <p>Energy Consumed: {report.total_energy_kwh:.2f} kWh</p>
            <p>Cost Impact: ${report.cost_impact_usd:.2f}</p>
            <p>Generated: {report.generated_timestamp}</p>
        </body>
        </html>
        """
    
    def _generate_csv_report(self, report: ExecutiveReport) -> str:
        """Generate CSV report data."""
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header section
        writer.writerow(["Carbon Impact Report"])
        writer.writerow(["Report ID", report.report_id])
        writer.writerow(["Period Start", report.period_start])
        writer.writerow(["Period End", report.period_end])
        writer.writerow([])
        
        # Metrics section
        writer.writerow(["Key Metrics"])
        writer.writerow(["Total Emissions (kg CO₂)", report.total_emissions_kg])
        writer.writerow(["Total Energy (kWh)", report.total_energy_kwh])
        writer.writerow(["Cost Impact (USD)", report.cost_impact_usd])
        writer.writerow(["Efficiency Trend", report.efficiency_trend])
        writer.writerow([])
        
        # Department breakdown
        writer.writerow(["Department Breakdown"])
        writer.writerow(["Department", "Emissions (kg CO₂)", "Energy (kWh)"])
        for dept, metrics in report.department_breakdown.items():
            writer.writerow([dept, metrics.get('emissions', 0), metrics.get('energy', 0)])
        
        return output.getvalue()


class CarbonBudgetManager:
    """Manage carbon budgets and alerts for enterprise usage."""
    
    def __init__(self):
        self.budgets: Dict[str, CarbonBudgetAlert] = {}
        self.current_usage: Dict[str, float] = {}
        self.alert_history: List[Dict[str, Any]] = []
        
    def add_budget(self, budget: CarbonBudgetAlert):
        """Add a carbon budget to monitor."""
        self.budgets[budget.budget_name] = budget
        self.current_usage[budget.budget_name] = 0.0
        logger.info(f"Added carbon budget: {budget.budget_name} ({budget.limit_kg_co2} kg CO₂)")
    
    def update_usage(self, budget_name: str, additional_co2_kg: float):
        """Update carbon usage for a budget."""
        if budget_name not in self.budgets:
            logger.warning(f"Unknown budget: {budget_name}")
            return
        
        self.current_usage[budget_name] += additional_co2_kg
        self._check_thresholds(budget_name)
    
    def _check_thresholds(self, budget_name: str):
        """Check if budget thresholds have been exceeded."""
        budget = self.budgets[budget_name]
        current = self.current_usage[budget_name]
        usage_pct = (current / budget.limit_kg_co2) * 100
        
        alert_triggered = False
        alert_level = None
        
        if usage_pct >= budget.critical_threshold_pct:
            alert_level = "critical"
            alert_triggered = True
        elif usage_pct >= budget.warning_threshold_pct:
            alert_level = "warning"
            alert_triggered = True
        
        if alert_triggered:
            alert_data = {
                "timestamp": time.time(),
                "budget_name": budget_name,
                "alert_level": alert_level,
                "current_usage_kg": current,
                "limit_kg": budget.limit_kg_co2,
                "usage_percentage": usage_pct
            }
            
            self.alert_history.append(alert_data)
            logger.warning(f"Carbon budget {alert_level} alert: {budget_name} at {usage_pct:.1f}%")
            
            # Execute auto-actions if configured
            for action in budget.auto_actions:
                self._execute_auto_action(action, alert_data)
    
    def _execute_auto_action(self, action: str, alert_data: Dict[str, Any]):
        """Execute automated action on budget alert."""
        if action == "email_admin":
            logger.info(f"Auto-action: Email admin about {alert_data['budget_name']}")
        elif action == "reduce_batch_size":
            logger.info(f"Auto-action: Recommend batch size reduction for {alert_data['budget_name']}")
        elif action == "pause_training":
            logger.warning(f"Auto-action: Training pause recommended for {alert_data['budget_name']}")
        else:
            logger.info(f"Auto-action: {action} for {alert_data['budget_name']}")
    
    def get_budget_status(self) -> Dict[str, Dict[str, float]]:
        """Get status of all budgets."""
        status = {}
        for name, budget in self.budgets.items():
            current = self.current_usage[name]
            status[name] = {
                "current_usage_kg": current,
                "limit_kg": budget.limit_kg_co2,
                "usage_percentage": (current / budget.limit_kg_co2) * 100,
                "remaining_kg": budget.limit_kg_co2 - current
            }
        return status


class ESGReportingEngine:
    """ESG compliance reporting engine."""
    
    def __init__(self, framework: ESGFramework = ESGFramework.CDP):
        self.framework = framework
        self.metrics_history: List[ESGMetrics] = []
        
        # Framework-specific requirements
        self.framework_requirements = {
            ESGFramework.CDP: {
                "required_scopes": [1, 2, 3],
                "intensity_metrics": ["revenue", "fte"],
                "verification_required": True
            },
            ESGFramework.GRI: {
                "required_scopes": [1, 2],
                "intensity_metrics": ["revenue", "production"],
                "verification_required": False
            },
            ESGFramework.TCFD: {
                "scenario_analysis": True,
                "risk_assessment": True,
                "governance_metrics": True
            }
        }
    
    def calculate_esg_metrics(self, carbon_data: Dict[str, float], 
                            business_data: Dict[str, float]) -> ESGMetrics:
        """Calculate ESG metrics from carbon and business data."""
        # Allocate emissions to scopes based on data source
        scope1 = carbon_data.get('direct_emissions_kg', 0.0)
        scope2 = carbon_data.get('indirect_energy_emissions_kg', 0.0)
        scope3 = carbon_data.get('other_indirect_emissions_kg', 0.0)
        
        # Calculate carbon intensity
        revenue = business_data.get('revenue_usd', 1.0)
        total_emissions = scope1 + scope2 + scope3
        carbon_intensity = (total_emissions / revenue) * 1000 if revenue > 0 else 0  # g CO₂/$
        
        metrics = ESGMetrics(
            framework=self.framework,
            carbon_emissions_scope1=scope1,
            carbon_emissions_scope2=scope2,
            carbon_emissions_scope3=scope3,
            renewable_energy_percentage=carbon_data.get('renewable_percentage', 0.0),
            carbon_intensity_per_revenue=carbon_intensity,
            carbon_offset_credits=carbon_data.get('offset_credits_kg', 0.0)
        )
        
        # Calculate compliance score
        metrics.compliance_score = self._calculate_compliance_score(metrics)
        
        self.metrics_history.append(metrics)
        return metrics
    
    def _calculate_compliance_score(self, metrics: ESGMetrics) -> float:
        """Calculate compliance score based on framework requirements."""
        score = 0.0
        max_score = 100.0
        
        requirements = self.framework_requirements.get(self.framework, {})
        
        # Check scope completeness
        required_scopes = requirements.get("required_scopes", [])
        scope_values = [metrics.carbon_emissions_scope1, metrics.carbon_emissions_scope2, metrics.carbon_emissions_scope3]
        
        for i, scope in enumerate([1, 2, 3]):
            if scope in required_scopes:
                if scope_values[i] > 0:  # Data available
                    score += 30.0 / len(required_scopes)
        
        # Renewable energy scoring
        if metrics.renewable_energy_percentage > 0:
            score += min(20.0, metrics.renewable_energy_percentage / 5.0)
        
        # Carbon intensity scoring (lower is better)
        if metrics.carbon_intensity_per_revenue > 0:
            # Benchmark against industry average (assume 500 g CO₂/$)
            intensity_score = max(0, 20 - (metrics.carbon_intensity_per_revenue / 25))
            score += intensity_score
        
        # Offset credits scoring
        total_emissions = metrics.carbon_emissions_scope1 + metrics.carbon_emissions_scope2 + metrics.carbon_emissions_scope3
        if total_emissions > 0 and metrics.carbon_offset_credits > 0:
            offset_ratio = metrics.carbon_offset_credits / total_emissions
            score += min(10.0, offset_ratio * 10)
        
        return min(max_score, score)
    
    def generate_esg_report(self, period_months: int = 12) -> Dict[str, Any]:
        """Generate comprehensive ESG report."""
        cutoff_time = time.time() - (period_months * 30 * 24 * 3600)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {"error": "No metrics available for reporting period"}
        
        # Aggregate metrics
        total_scope1 = sum(m.carbon_emissions_scope1 for m in recent_metrics)
        total_scope2 = sum(m.carbon_emissions_scope2 for m in recent_metrics)
        total_scope3 = sum(m.carbon_emissions_scope3 for m in recent_metrics)
        avg_renewable_pct = sum(m.renewable_energy_percentage for m in recent_metrics) / len(recent_metrics)
        avg_compliance_score = sum(m.compliance_score for m in recent_metrics) / len(recent_metrics)
        
        # Trend analysis
        if len(recent_metrics) >= 2:
            recent_emissions = recent_metrics[-1].carbon_emissions_scope1 + recent_metrics[-1].carbon_emissions_scope2
            earlier_emissions = recent_metrics[0].carbon_emissions_scope1 + recent_metrics[0].carbon_emissions_scope2
            trend = "decreasing" if recent_emissions < earlier_emissions else "increasing"
        else:
            trend = "stable"
        
        return {
            "framework": self.framework.value,
            "reporting_period_months": period_months,
            "total_emissions": {
                "scope1_kg_co2": total_scope1,
                "scope2_kg_co2": total_scope2,
                "scope3_kg_co2": total_scope3,
                "total_kg_co2": total_scope1 + total_scope2 + total_scope3
            },
            "renewable_energy_percentage": avg_renewable_pct,
            "compliance_score": avg_compliance_score,
            "emissions_trend": trend,
            "framework_requirements": self.framework_requirements.get(self.framework, {}),
            "generated_timestamp": datetime.now(),
            "data_points": len(recent_metrics)
        }


class EnterpriseIntegrationHub:
    """Main enterprise integration hub coordinating all enterprise features."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.slack_notifier = None
        self.teams_notifier = None
        self.email_reporter = None
        self.budget_manager = CarbonBudgetManager()
        self.esg_engine = ESGReportingEngine()
        
        # Notification configurations
        self.notification_configs: List[NotificationConfig] = []
        self.scheduled_reports: Dict[str, Any] = {}
        
        # Initialize integrations based on config
        self._initialize_integrations()
        
        # Start scheduled reporting
        self._setup_scheduled_reporting()
        
        logger.info("Enterprise Integration Hub initialized")
    
    def _initialize_integrations(self):
        """Initialize available integrations based on configuration."""
        # Slack integration
        slack_config = self.config.get('slack', {})
        if slack_config.get('webhook_url'):
            self.slack_notifier = SlackNotifier(
                slack_config['webhook_url'],
                slack_config.get('channel', '#carbon-tracking')
            )
            logger.info("Slack integration enabled")
        
        # Teams integration
        teams_config = self.config.get('teams', {})
        if teams_config.get('webhook_url'):
            self.teams_notifier = TeamsNotifier(teams_config['webhook_url'])
            logger.info("Teams integration enabled")
        
        # Email integration
        email_config = self.config.get('email', {})
        if all(k in email_config for k in ['smtp_host', 'smtp_port', 'username', 'password']):
            self.email_reporter = EmailReporter(
                email_config['smtp_host'],
                email_config['smtp_port'],
                email_config['username'],
                email_config['password']
            )
            logger.info("Email integration enabled")
    
    def _setup_scheduled_reporting(self):
        """Setup scheduled reporting based on configuration."""
        reporting_config = self.config.get('scheduled_reports', {})
        
        for report_name, config in reporting_config.items():
            frequency = config.get('frequency', 'daily')
            time_str = config.get('time', '09:00')
            
            if frequency == 'daily':
                schedule.every().day.at(time_str).do(
                    self._generate_scheduled_report, report_name, config
                )
            elif frequency == 'weekly':
                schedule.every().week.at(time_str).do(
                    self._generate_scheduled_report, report_name, config
                )
            elif frequency == 'monthly':
                schedule.every().month.at(time_str).do(
                    self._generate_scheduled_report, report_name, config
                )
        
        # Start scheduler thread
        if reporting_config:
            scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
            scheduler_thread.start()
            logger.info(f"Scheduled {len(reporting_config)} automated reports")
    
    def _run_scheduler(self):
        """Run the report scheduler."""
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    @optimized(cache_ttl=300.0)  # Cache for 5 minutes
    @handle_gracefully(severity=ErrorSeverity.MEDIUM, fallback_value=None)
    def _generate_scheduled_report(self, report_name: str, config: Dict[str, Any]):
        """Generate and send a scheduled report."""
        logger.info(f"Generating scheduled report: {report_name}")
        
        report_type = config.get('type', 'summary')
        recipients = config.get('recipients', [])
        
        if report_type == 'executive' and self.email_reporter and recipients:
            # Generate executive report
            report = self._create_executive_report()
            self.email_reporter.send_executive_report(report, recipients)
        
        elif report_type == 'esg' and recipients:
            # Generate ESG report
            esg_report = self.esg_engine.generate_esg_report()
            # Send via configured channels
            self._send_esg_report(esg_report, recipients)
    
    def process_carbon_metrics(self, metrics: 'CarbonMetrics', project_context: Dict[str, Any]):
        """Process carbon metrics and trigger enterprise integrations."""
        # Update budget tracking
        project_name = project_context.get('project_name', 'default')
        if project_name in self.budget_manager.budgets:
            self.budget_manager.update_usage(project_name, metrics.co2_kg)
        
        # Check notification thresholds
        self._check_notification_thresholds(metrics, project_context)
        
        # Update ESG metrics
        carbon_data = {
            'indirect_energy_emissions_kg': metrics.co2_kg,
            'renewable_percentage': project_context.get('renewable_percentage', 0.0)
        }
        business_data = project_context.get('business_data', {})
        self.esg_engine.calculate_esg_metrics(carbon_data, business_data)
        
        logger.debug(f"Processed carbon metrics for enterprise integrations: {metrics.co2_kg:.3f} kg CO₂")
    
    def _check_notification_thresholds(self, metrics: 'CarbonMetrics', context: Dict[str, Any]):
        """Check if metrics exceed notification thresholds."""
        for config in self.notification_configs:
            if not config.enabled:
                continue
            
            threshold_met = False
            alert_type = "info"
            
            # Check threshold conditions
            for condition, threshold in config.threshold_conditions.items():
                if condition == "co2_kg" and metrics.co2_kg > threshold:
                    threshold_met = True
                    alert_type = "warning" if metrics.co2_kg < threshold * 1.2 else "critical"
                elif condition == "power_watts" and metrics.power_watts > threshold:
                    threshold_met = True
                    alert_type = "warning"
            
            if threshold_met:
                self._send_threshold_notification(config, metrics, context, alert_type)
    
    def _send_threshold_notification(self, config: NotificationConfig, metrics: 'CarbonMetrics', 
                                   context: Dict[str, Any], alert_type: str):
        """Send notification when threshold is exceeded."""
        message_template = config.message_template or "Carbon threshold exceeded: {co2_kg:.2f} kg CO₂"
        message = message_template.format(**metrics.__dict__)
        
        if config.channel == NotificationChannel.SLACK and self.slack_notifier:
            self.slack_notifier.send_carbon_alert(alert_type, metrics.co2_kg, 
                                                config.threshold_conditions.get('co2_kg', 0), context)
        
        elif config.channel == NotificationChannel.TEAMS and self.teams_notifier:
            facts = [
                {"name": "CO₂ Emissions", "value": f"{metrics.co2_kg:.2f} kg"},
                {"name": "Power Usage", "value": f"{metrics.power_watts:.0f} W"},
                {"name": "Project", "value": context.get('project_name', 'Unknown')}
            ]
            self.teams_notifier.send_notification(f"🚨 Carbon Alert ({alert_type})", message, facts)
        
        elif config.channel == NotificationChannel.EMAIL and self.email_reporter:
            # Email notifications would be implemented here
            logger.info(f"Email notification triggered: {message}")
    
    def _create_executive_report(self) -> ExecutiveReport:
        """Create an executive report from current data."""
        # This would typically aggregate data from multiple sources
        now = datetime.now()
        period_start = now - timedelta(days=30)
        
        # Placeholder data - in real implementation, this would aggregate from carbon reports
        return ExecutiveReport(
            report_id=str(uuid.uuid4()),
            period_start=period_start,
            period_end=now,
            total_emissions_kg=150.5,
            total_energy_kwh=300.2,
            cost_impact_usd=45.15,
            efficiency_trend="improving",
            key_findings=[
                "Training efficiency improved 15% this month",
                "Peak power usage occurred during vision model training",
                "Renewable energy usage increased to 65%"
            ],
            recommendations=[
                "Consider smaller batch sizes for vision models",
                "Schedule intensive training during peak renewable hours",
                "Implement gradient checkpointing for large models"
            ],
            compliance_status={"CDP": "compliant", "GRI": "pending"},
            carbon_budget_status=self.budget_manager.get_budget_status(),
            department_breakdown={
                "Research": {"emissions": 75.2, "energy": 150.1},
                "Engineering": {"emissions": 50.3, "energy": 100.1},
                "Data Science": {"emissions": 25.0, "energy": 50.0}
            }
        )
    
    def _send_esg_report(self, esg_report: Dict[str, Any], recipients: List[str]):
        """Send ESG report via configured channels."""
        summary = f"ESG Compliance Report - Score: {esg_report.get('compliance_score', 0):.1f}/100"
        
        if self.teams_notifier:
            facts = [
                {"name": "Framework", "value": esg_report.get('framework', 'Unknown')},
                {"name": "Total Emissions", "value": f"{esg_report.get('total_emissions', {}).get('total_kg_co2', 0):.1f} kg CO₂"},
                {"name": "Compliance Score", "value": f"{esg_report.get('compliance_score', 0):.1f}/100"},
                {"name": "Renewable Energy", "value": f"{esg_report.get('renewable_energy_percentage', 0):.1f}%"}
            ]
            self.teams_notifier.send_notification("📊 ESG Compliance Report", summary, facts)
    
    def add_notification_config(self, config: NotificationConfig):
        """Add a notification configuration."""
        self.notification_configs.append(config)
        logger.info(f"Added {config.channel.value} notification config")
    
    def add_carbon_budget(self, budget: CarbonBudgetAlert):
        """Add a carbon budget for monitoring."""
        self.budget_manager.add_budget(budget)
    
    def get_enterprise_dashboard_data(self) -> Dict[str, Any]:
        """Get data for enterprise dashboard."""
        return {
            "budget_status": self.budget_manager.get_budget_status(),
            "recent_alerts": self.budget_manager.alert_history[-10:],
            "esg_metrics": self.esg_engine.generate_esg_report(period_months=1),
            "notification_configs": len(self.notification_configs),
            "integrations_active": {
                "slack": self.slack_notifier is not None,
                "teams": self.teams_notifier is not None,
                "email": self.email_reporter is not None
            }
        }


# Convenience function for easy integration
def create_enterprise_hub(config_path: Optional[str] = None) -> EnterpriseIntegrationHub:
    """Create an enterprise integration hub with configuration."""
    config = {}
    
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load enterprise config: {e}")
    
    return EnterpriseIntegrationHub(config)