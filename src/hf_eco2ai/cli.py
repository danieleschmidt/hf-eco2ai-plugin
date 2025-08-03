"""Command line interface for HF Eco2AI Plugin."""

import click
import json
import sys
from pathlib import Path
from typing import Optional

from .config import CarbonConfig
from .models import CarbonReport
from .monitoring import EnergyTracker


@click.group()
@click.version_option()
def cli():
    """HF Eco2AI Plugin: Carbon tracking for ML training."""
    pass


@cli.command()
@click.option('--report', '-r', required=True, help='Path to carbon report JSON file')
@click.option('--max-co2', '--budget', type=float, required=True, help='Maximum CO₂ budget in kg')
@click.option('--format', 'output_format', default='text', type=click.Choice(['text', 'json']),
              help='Output format')
def check_budget(report: str, max_co2: float, output_format: str):
    """Check if training exceeded carbon budget."""
    try:
        if not Path(report).exists():
            click.echo(f"Error: Report file not found: {report}", err=True)
            sys.exit(1)
        
        # Load carbon report
        report_data = json.loads(Path(report).read_text())
        current_co2 = report_data.get('summary', {}).get('total_co2_kg', 0)
        
        budget_exceeded = current_co2 > max_co2
        percentage_used = (current_co2 / max_co2) * 100 if max_co2 > 0 else 0
        
        if output_format == 'json':
            result = {
                "budget_kg": max_co2,
                "actual_kg": current_co2,
                "exceeded": budget_exceeded,
                "percentage_used": percentage_used,
                "remaining_kg": max(0, max_co2 - current_co2)
            }
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"Carbon Budget Check")
            click.echo(f"==================")
            click.echo(f"Budget: {max_co2:.2f} kg CO₂")
            click.echo(f"Actual: {current_co2:.2f} kg CO₂")
            click.echo(f"Used: {percentage_used:.1f}%")
            
            if budget_exceeded:
                click.echo(f"❌ BUDGET EXCEEDED by {current_co2 - max_co2:.2f} kg", err=True)
                sys.exit(1)
            else:
                click.echo(f"✅ Within budget ({max_co2 - current_co2:.2f} kg remaining)")
        
    except Exception as e:
        click.echo(f"Error checking budget: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--config', '-c', help='Path to configuration file')
@click.option('--project', '-p', default='validation', help='Project name for validation')
@click.option('--dry-run', is_flag=True, help='Run validation without starting actual tracking')
def validate(config: Optional[str], project: str, dry_run: bool):
    """Validate carbon tracking setup and configuration."""
    try:
        # Load configuration
        if config:
            carbon_config = CarbonConfig.from_json(config)
        else:
            carbon_config = CarbonConfig(project_name=project)
        
        click.echo(f"Validating carbon tracking setup for project: {project}")
        click.echo("=" * 50)
        
        # Check dependencies
        missing_deps = carbon_config.validate_environment()
        if missing_deps:
            click.echo("❌ Missing dependencies:")
            for dep in missing_deps:
                click.echo(f"   - {dep}")
            click.echo("\nInstall missing dependencies with:")
            click.echo(f"   pip install {' '.join(missing_deps)}")
            sys.exit(1)
        else:
            click.echo("✅ All dependencies available")
        
        # Test energy tracking
        energy_tracker = EnergyTracker(
            gpu_ids=carbon_config.gpu_ids,
            country=carbon_config.country,
            region=carbon_config.region
        )
        
        if energy_tracker.is_available():
            click.echo("✅ GPU monitoring available")
            
            # Test metrics collection
            if not dry_run:
                click.echo("Testing metric collection...")
                energy_tracker.start_tracking()
                
                import time
                time.sleep(2)  # Collect for 2 seconds
                
                power, energy, co2 = energy_tracker.get_current_consumption()
                energy_tracker.stop_tracking()
                
                click.echo(f"   Power: {power:.1f} W")
                click.echo(f"   Energy: {energy:.6f} kWh")
                click.echo(f"   CO₂: {co2:.6f} kg")
            else:
                click.echo("   (Skipped - dry run)")
        else:
            click.echo("⚠️  GPU monitoring not available (estimation mode will be used)")
        
        # Test carbon intensity
        carbon_intensity = energy_tracker.carbon_provider.get_carbon_intensity()
        renewable_pct = energy_tracker.carbon_provider.get_renewable_percentage()
        
        click.echo(f"✅ Carbon intensity data: {carbon_intensity:.0f} g CO₂/kWh")
        click.echo(f"✅ Renewable percentage: {renewable_pct:.1f}%")
        
        # Test Prometheus export if enabled
        if carbon_config.export_prometheus:
            try:
                from .exporters import PrometheusExporter
                exporter = PrometheusExporter(carbon_config.prometheus_port)
                click.echo(f"✅ Prometheus exporter: {exporter.get_metrics_url()}")
            except Exception as e:
                click.echo(f"❌ Prometheus setup failed: {e}")
        
        # Test output paths
        if carbon_config.save_report:
            report_path = Path(carbon_config.report_path)
            if report_path.parent.exists() or report_path.parent == Path('.'):
                click.echo(f"✅ Report output path: {carbon_config.report_path}")
            else:
                click.echo(f"❌ Report output directory does not exist: {report_path.parent}")
        
        click.echo("\n🎉 Validation completed successfully!")
        
    except Exception as e:
        click.echo(f"❌ Validation failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--report', '-r', required=True, help='Path to carbon report JSON file')
@click.option('--format', 'output_format', default='text', type=click.Choice(['text', 'json', 'csv']),
              help='Output format')
@click.option('--output', '-o', help='Output file path (default: stdout)')
def report(report: str, output_format: str, output: Optional[str]):
    """Generate formatted carbon report from tracking data."""
    try:
        if not Path(report).exists():
            click.echo(f"Error: Report file not found: {report}", err=True)
            sys.exit(1)
        
        # Load report
        report_data = json.loads(Path(report).read_text())
        carbon_report = CarbonReport()
        
        # Reconstruct summary from data
        summary_data = report_data.get('summary', {})
        carbon_report.summary.total_energy_kwh = summary_data.get('total_energy_kwh', 0)
        carbon_report.summary.total_co2_kg = summary_data.get('total_co2_kg', 0)
        carbon_report.summary.total_duration_hours = summary_data.get('total_duration_hours', 0)
        carbon_report.summary.average_power_watts = summary_data.get('average_power_watts', 0)
        carbon_report.summary.total_samples = summary_data.get('total_samples', 0)
        carbon_report.summary.calculate_equivalents()
        
        # Generate output
        if output_format == 'text':
            content = carbon_report.summary_text()
        elif output_format == 'json':
            content = json.dumps(report_data, indent=2)
        elif output_format == 'csv':
            # Generate CSV summary
            content = "metric,value\n"
            content += f"total_energy_kwh,{carbon_report.summary.total_energy_kwh}\n"
            content += f"total_co2_kg,{carbon_report.summary.total_co2_kg}\n"
            content += f"duration_hours,{carbon_report.summary.total_duration_hours}\n"
            content += f"average_power_watts,{carbon_report.summary.average_power_watts}\n"
            content += f"equivalent_km_driven,{carbon_report.summary.equivalent_km_driven}\n"
        
        # Output to file or stdout
        if output:
            Path(output).write_text(content)
            click.echo(f"Report written to: {output}")
        else:
            click.echo(content)
            
    except Exception as e:
        click.echo(f"Error generating report: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--project', '-p', default='optimization-test', help='Project name')
@click.option('--model-size', type=click.Choice(['small', 'medium', 'large']), default='medium',
              help='Model size category')
@click.option('--duration', type=int, default=60, help='Estimated training duration in minutes')
@click.option('--format', 'output_format', default='text', type=click.Choice(['text', 'json']),
              help='Output format')
def optimize(project: str, model_size: str, duration: int, output_format: str):
    """Get optimization recommendations for reducing carbon footprint."""
    try:
        # Model size to parameter mapping (rough estimates)
        model_params = {
            'small': 100_000_000,    # 100M parameters
            'medium': 1_000_000_000, # 1B parameters  
            'large': 10_000_000_000  # 10B parameters
        }
        
        params = model_params[model_size]
        duration_hours = duration / 60
        
        # Estimate power consumption based on model size
        base_power = {
            'small': 200,   # 200W
            'medium': 500,  # 500W
            'large': 1500   # 1500W
        }
        
        estimated_power = base_power[model_size]
        estimated_energy = estimated_power * duration_hours / 1000  # kWh
        estimated_co2 = estimated_energy * 400 / 1000  # kg CO₂ (using 400g/kWh average)
        
        # Generate recommendations
        recommendations = []
        
        # Mixed precision training
        recommendations.append({
            "title": "Enable Mixed Precision Training",
            "description": "Use FP16 or BF16 to reduce memory usage and energy by 30-40%",
            "potential_reduction_percent": 35,
            "implementation": "Add fp16=True to TrainingArguments",
            "energy_savings_kwh": estimated_energy * 0.35,
            "co2_savings_kg": estimated_co2 * 0.35
        })
        
        # Gradient checkpointing
        if model_size in ['medium', 'large']:
            recommendations.append({
                "title": "Enable Gradient Checkpointing", 
                "description": "Trade computation for memory to enable larger batches",
                "potential_reduction_percent": 20,
                "implementation": "Add gradient_checkpointing=True to TrainingArguments",
                "energy_savings_kwh": estimated_energy * 0.20,
                "co2_savings_kg": estimated_co2 * 0.20
            })
        
        # Low-carbon scheduling
        recommendations.append({
            "title": "Schedule Training During Low-Carbon Hours",
            "description": "Run training when grid carbon intensity is lowest",
            "potential_reduction_percent": 25,
            "implementation": "Schedule training between 11 PM - 6 AM",
            "energy_savings_kwh": 0,  # Same energy, lower carbon
            "co2_savings_kg": estimated_co2 * 0.25
        })
        
        # Model optimization
        if model_size == 'large':
            recommendations.append({
                "title": "Consider Model Pruning",
                "description": "Remove redundant parameters to reduce computation",
                "potential_reduction_percent": 30,
                "implementation": "Apply structured pruning during training", 
                "energy_savings_kwh": estimated_energy * 0.30,
                "co2_savings_kg": estimated_co2 * 0.30
            })
        
        # Format output
        if output_format == 'json':
            result = {
                "project": project,
                "model_size": model_size,
                "estimated_metrics": {
                    "power_watts": estimated_power,
                    "energy_kwh": estimated_energy,
                    "co2_kg": estimated_co2,
                    "duration_hours": duration_hours
                },
                "recommendations": recommendations,
                "total_potential_savings": {
                    "energy_kwh": sum(r["energy_savings_kwh"] for r in recommendations),
                    "co2_kg": sum(r["co2_savings_kg"] for r in recommendations)
                }
            }
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"Carbon Optimization Recommendations")
            click.echo(f"===================================")
            click.echo(f"Project: {project}")
            click.echo(f"Model size: {model_size} ({params:,} parameters)")
            click.echo(f"Duration: {duration_hours:.1f} hours")
            click.echo(f"")
            click.echo(f"Estimated Impact:")
            click.echo(f"  Power: {estimated_power} W")
            click.echo(f"  Energy: {estimated_energy:.2f} kWh")
            click.echo(f"  CO₂: {estimated_co2:.2f} kg")
            click.echo(f"")
            click.echo(f"Recommendations:")
            
            for i, rec in enumerate(recommendations, 1):
                click.echo(f"")
                click.echo(f"{i}. {rec['title']}")
                click.echo(f"   {rec['description']}")
                click.echo(f"   Potential reduction: {rec['potential_reduction_percent']}%")
                click.echo(f"   Energy savings: {rec['energy_savings_kwh']:.2f} kWh")
                click.echo(f"   CO₂ savings: {rec['co2_savings_kg']:.2f} kg")
                click.echo(f"   Implementation: {rec['implementation']}")
            
            total_energy_savings = sum(r["energy_savings_kwh"] for r in recommendations)
            total_co2_savings = sum(r["co2_savings_kg"] for r in recommendations)
            
            click.echo(f"")
            click.echo(f"Total Potential Savings:")
            click.echo(f"  Energy: {total_energy_savings:.2f} kWh ({total_energy_savings/estimated_energy*100:.1f}%)")
            click.echo(f"  CO₂: {total_co2_savings:.2f} kg ({total_co2_savings/estimated_co2*100:.1f}%)")
            
    except Exception as e:
        click.echo(f"Error generating optimization recommendations: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--config', '-c', help='Path to configuration file')
@click.option('--output', '-o', default='config.json', help='Output configuration file')
def config(config: Optional[str], output: str):
    """Generate or validate configuration file."""
    try:
        if config:
            # Validate existing config
            carbon_config = CarbonConfig.from_json(config)
            click.echo(f"✅ Configuration file {config} is valid")
            
            # Show current settings
            click.echo("\nCurrent configuration:")
            config_dict = carbon_config.to_dict()
            for key, value in config_dict.items():
                if not key.startswith('_'):
                    click.echo(f"  {key}: {value}")
        else:
            # Generate default config
            carbon_config = CarbonConfig()
            carbon_config.to_json(output)
            click.echo(f"✅ Generated default configuration: {output}")
            click.echo("Edit the file to customize settings for your project.")
            
    except Exception as e:
        click.echo(f"Error with configuration: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()