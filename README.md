# 🌱 hf-eco2ai-plugin

**Carbon emission tracking for HuggingFace Trainer runs.**

Estimates GPU energy consumption during training and converts it to CO₂ emissions
using real regional grid carbon intensity data. Drop-in `TrainerCallback` with zero
required dependencies — works with or without a real GPU.

---

## Why?

Training ML models has a real carbon footprint. A single large model training run can
emit as much CO₂ as a transatlantic flight. Switching from a coal-heavy region (e.g.
Mumbai: 708 gCO₂/kWh) to a clean one (e.g. Stockholm: 13 gCO₂/kWh) can reduce
emissions by **54×** for identical workloads — before any model or hardware changes.

This plugin makes that visible.

---

## Install

```bash
pip install hf-eco2ai-plugin
```

No hard dependencies. For HuggingFace Trainer integration:

```bash
pip install hf-eco2ai-plugin[transformers]
```

---

## Quickstart

### With HuggingFace Trainer

```python
from hf_eco2ai import Eco2AICallback

trainer = Trainer(
    model=model,
    args=training_args,
    callbacks=[Eco2AICallback(region="eu-west-3", gpu_tdp_w=400)],
)
trainer.train()
# Prints emissions report at the end of training
```

### Standalone (no HuggingFace needed)

```python
from hf_eco2ai import Eco2AICallback

cb = Eco2AICallback(region="us-west-2", gpu_tdp_w=300)
cb.on_train_begin()

for step in range(num_steps):
    # ... your training logic ...
    cb.on_step_end(step=step)

report = cb.on_train_end()
print(f"Total CO₂: {report.total_co2_kg * 1000:.2f} gCO₂")
print(f"Equivalent car travel: {report.equivalent_car_km:.4f} km")
```

### Lower-level CarbonTracker API

```python
from hf_eco2ai import CarbonTracker

tracker = CarbonTracker(
    gpu_tdp_w=350,       # GPU thermal design power in watts
    region="eu-west-3",  # or any AWS/GCP/Azure region, country code
    utilization=0.8,     # assumed GPU utilization
    capture_steps=True,  # record per-step snapshots
)

tracker.start()
for step in range(steps):
    do_training_step()
    tracker.step(step)

report = tracker.stop()
print(report)
```

---

## Supported Regions

Over 30 cloud regions and country codes are built in:

| Region ID | Location | gCO₂/kWh | Renewable % |
|-----------|----------|-----------|-------------|
| `eu-north-1` | Stockholm (SE) | 13 | 98% |
| `eu-west-3` | Paris (FR) | 58 | 91% |
| `us-west-2` | Oregon (US) | 130 | 68% |
| `eu-west-1` | Ireland | 290 | 40% |
| `us-east-1` | N. Virginia | 379 | 28% |
| `ap-south-1` | Mumbai (IN) | 708 | 8% |
| `af-south-1` | Cape Town (ZA) | 928 | 6% |

AWS, GCP, and Azure region IDs are all supported. Full list via:

```python
from hf_eco2ai import RegionalGrid
grid = RegionalGrid()
for rid, r in grid.list_regions().items():
    print(f"{rid:20} {r.carbon_intensity_g_per_kwh:>6.0f} gCO₂/kWh")
```

Custom regions:

```python
from hf_eco2ai import RegionalGrid, GridRegion

grid = RegionalGrid(custom_regions={
    "my-dc-1": GridRegion("My Datacenter", "US", carbon_intensity_g_per_kwh=45.0, renewable_pct=88.0)
})
```

---

## EmissionsReport

Every tracking run returns an `EmissionsReport`:

```python
@dataclass
class EmissionsReport:
    total_energy_kwh: float      # Total electrical energy consumed
    total_co2_kg: float          # Total CO₂-equivalent emissions
    equivalent_car_km: float     # Driving distance with equivalent emissions
    total_steps: int             # Training steps completed
    emissions_per_step: float    # Average kg CO₂ per step
    duration_hours: float        # Wall-clock training duration
    region: str                  # Grid region used
    carbon_intensity_g_per_kwh: float
    gpu_tdp_w: float
    snapshots: list[StepSnapshot]  # Per-step data (if capture_steps=True)
```

---

## Callback Options

```python
Eco2AICallback(
    region="us-east-1",        # Grid region (default: us-east-1)
    gpu_tdp_w=300.0,           # GPU TDP in watts (default: 300 W)
    utilization=0.8,           # GPU utilization fraction (default: 0.8)
    capture_steps=False,       # Record per-step snapshots (default: False)
    power_override_w=None,     # Live power reading from NVML (optional)
    log_every_n_steps=0,       # Log progress every N steps (0 = off)
)
```

**Live GPU power via pynvml:**

```python
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
live_watts = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW → W

cb = Eco2AICallback(region="us-east-1", power_override_w=live_watts)
```

---

## Carbon Intensity Data

Data sourced from:
- IEA (2023) electricity carbon intensity by country
- Cloud provider sustainability reports (AWS, GCP, Azure)
- electricityMap.org regional averages

Values represent annual grid averages. Real-time intensity varies — for
high-accuracy tracking, integrate with the [electricityMap API](https://www.electricitymap.org/api).

---

## Demo

```bash
python demo.py
```

Simulates a 100-step training run, compares 5 cloud regions, and shows
the lower-level `CarbonTracker` API. No GPU or HuggingFace required.

---

## Development

```bash
git clone https://github.com/danieleschmidt/hf-eco2ai-plugin
cd hf-eco2ai-plugin
pip install -e ".[dev]"
pytest tests/
```

---

## License

MIT — see [LICENSE](LICENSE).

---

*Track it. Report it. Train greener.*
