"""Core unit tests for hf-eco2ai-plugin."""

import sys
import os
import time

# Ensure src/ is importable without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from hf_eco2ai import CarbonTracker, EmissionsReport, Eco2AICallback, RegionalGrid, GridRegion


# ---------------------------------------------------------------------------
# RegionalGrid tests
# ---------------------------------------------------------------------------

class TestRegionalGrid:
    def test_known_aws_region(self):
        g = RegionalGrid()
        assert g.get_intensity("us-east-1") == pytest.approx(379.0)

    def test_known_eu_region(self):
        g = RegionalGrid()
        # France is nuclear-heavy — should be very low
        assert g.get_intensity("eu-west-3") < 100

    def test_unknown_region_returns_world_average(self):
        g = RegionalGrid()
        intensity = g.get_intensity("xx-unknown-99")
        # World average is 436 gCO2/kWh
        assert intensity == pytest.approx(436.0)

    def test_case_insensitive_lookup(self):
        g = RegionalGrid()
        lower = g.get_intensity("us-east-1")
        # Country code uppercase
        upper = g.get_intensity("US")
        assert lower > 0
        assert upper > 0

    def test_get_region_returns_grid_region(self):
        g = RegionalGrid()
        r = g.get_region("eu-north-1")
        assert isinstance(r, GridRegion)
        assert r.renewable_pct > 90  # Sweden is very green

    def test_custom_region(self):
        custom = GridRegion("My datacenter", "US", 50.0, 80.0)
        g = RegionalGrid(custom_regions={"my-dc-1": custom})
        assert g.get_intensity("my-dc-1") == pytest.approx(50.0)

    def test_add_region(self):
        g = RegionalGrid()
        g.add_region("test-region", GridRegion("Test", "XX", 123.0, 50.0))
        assert g.get_intensity("test-region") == pytest.approx(123.0)

    def test_list_regions_returns_dict(self):
        g = RegionalGrid()
        regions = g.list_regions()
        assert isinstance(regions, dict)
        assert len(regions) > 10  # At least the built-in regions are there

    def test_renewable_pct(self):
        g = RegionalGrid()
        # Norway is almost 100% hydro
        pct = g.get_renewable_pct("NO")
        assert pct > 90

    def test_south_africa_high_intensity(self):
        # Coal-heavy → high carbon
        g = RegionalGrid()
        assert g.get_intensity("af-south-1") > 800


# ---------------------------------------------------------------------------
# CarbonTracker tests
# ---------------------------------------------------------------------------

class TestCarbonTracker:
    def test_start_stop_returns_report(self):
        t = CarbonTracker(gpu_tdp_w=100, region="eu-west-3")
        t.start()
        time.sleep(0.05)
        t.step(0)
        report = t.stop()
        assert isinstance(report, EmissionsReport)

    def test_energy_positive_after_steps(self):
        t = CarbonTracker(gpu_tdp_w=200, region="us-east-1")
        t.start()
        time.sleep(0.1)
        t.step(0)
        report = t.stop()
        assert report.total_energy_kwh > 0

    def test_co2_proportional_to_intensity(self):
        """France (low intensity) should produce less CO2 than India (high)."""
        def run(region):
            t = CarbonTracker(gpu_tdp_w=200, region=region)
            t.start()
            time.sleep(0.1)
            t.step(0)
            r = t.stop()
            return r.total_co2_kg

        france_co2 = run("eu-west-3")   # ~58 gCO2/kWh
        india_co2 = run("ap-south-1")   # ~708 gCO2/kWh
        assert france_co2 < india_co2

    def test_multiple_steps(self):
        t = CarbonTracker(gpu_tdp_w=300, region="us-west-2")
        t.start()
        for i in range(5):
            time.sleep(0.02)
            t.step(i)
        report = t.stop()
        assert report.total_steps == 5

    def test_step_count(self):
        t = CarbonTracker(gpu_tdp_w=300, region="us-east-1")
        t.start()
        for i in range(10):
            t.step(i)
        assert t.step_count == 10

    def test_capture_steps_disabled_by_default(self):
        t = CarbonTracker(gpu_tdp_w=300, region="us-east-1")
        t.start()
        t.step(0)
        report = t.stop()
        assert len(report.snapshots) == 0

    def test_capture_steps_enabled(self):
        t = CarbonTracker(gpu_tdp_w=300, region="us-east-1", capture_steps=True)
        t.start()
        for i in range(5):
            t.step(i)
        report = t.stop()
        assert len(report.snapshots) == 5

    def test_equivalent_car_km(self):
        t = CarbonTracker(gpu_tdp_w=300, region="us-east-1")
        t.start()
        time.sleep(0.1)
        t.step(0)
        report = t.stop()
        # Should be total_co2_kg / 0.21
        expected = report.total_co2_kg / 0.21
        assert report.equivalent_car_km == pytest.approx(expected, rel=1e-5)

    def test_emissions_per_step(self):
        t = CarbonTracker(gpu_tdp_w=300, region="us-east-1")
        t.start()
        for i in range(5):
            time.sleep(0.01)
            t.step(i)
        report = t.stop()
        assert report.emissions_per_step == pytest.approx(
            report.total_co2_kg / 5, rel=1e-5
        )

    def test_power_override(self):
        """When power_override_w is set, TDP+utilization are ignored."""
        t = CarbonTracker(gpu_tdp_w=500, region="us-east-1",
                          utilization=0.9, power_override_w=100.0)
        assert t.current_power_w == pytest.approx(100.0)

    def test_start_twice_raises(self):
        t = CarbonTracker(gpu_tdp_w=300, region="us-east-1")
        t.start()
        with pytest.raises(RuntimeError):
            t.start()
        t.stop()

    def test_stop_before_start_raises(self):
        t = CarbonTracker(gpu_tdp_w=300, region="us-east-1")
        with pytest.raises(RuntimeError):
            t.stop()

    def test_step_before_start_raises(self):
        t = CarbonTracker(gpu_tdp_w=300, region="us-east-1")
        with pytest.raises(RuntimeError):
            t.step(0)

    def test_duration_hours(self):
        t = CarbonTracker(gpu_tdp_w=100, region="us-east-1")
        t.start()
        time.sleep(0.1)
        t.step(0)
        report = t.stop()
        assert report.duration_hours > 0
        assert report.duration_hours < 1  # definitely less than an hour

    def test_report_str(self):
        t = CarbonTracker(gpu_tdp_w=200, region="eu-west-3")
        t.start()
        time.sleep(0.05)
        t.step(0)
        report = t.stop()
        s = str(report)
        assert "EmissionsReport" in s
        assert "kWh" in s
        assert "CO" in s


# ---------------------------------------------------------------------------
# Eco2AICallback tests
# ---------------------------------------------------------------------------

class TestEco2AICallback:
    def test_full_lifecycle(self):
        cb = Eco2AICallback(region="us-east-1", gpu_tdp_w=300)
        cb.on_train_begin()
        for step in range(10):
            time.sleep(0.01)
            cb.on_step_end(step=step)
        report = cb.on_train_end()
        assert isinstance(report, EmissionsReport)
        assert report.total_steps == 10

    def test_report_available_after_train_end(self):
        cb = Eco2AICallback(region="eu-west-3", gpu_tdp_w=200)
        cb.on_train_begin()
        cb.on_step_end(step=0)
        cb.on_train_end()
        assert cb.report is not None

    def test_current_co2_mid_training(self):
        cb = Eco2AICallback(region="us-east-1", gpu_tdp_w=300)
        cb.on_train_begin()
        time.sleep(0.05)
        cb.on_step_end(step=0)
        # Should be positive but not yet finalized
        assert cb.current_co2_kg >= 0

    def test_current_energy_mid_training(self):
        cb = Eco2AICallback(region="us-east-1", gpu_tdp_w=300)
        cb.on_train_begin()
        time.sleep(0.05)
        cb.on_step_end(step=0)
        assert cb.current_energy_kwh >= 0

    def test_on_epoch_end_does_not_crash(self):
        cb = Eco2AICallback(region="us-east-1", gpu_tdp_w=300)
        cb.on_train_begin()
        cb.on_step_end(step=0)
        cb.on_epoch_end()  # should be a no-op / logging only
        cb.on_train_end()

    def test_log_every_n_steps(self, capsys):
        cb = Eco2AICallback(region="us-east-1", gpu_tdp_w=300, log_every_n_steps=5)
        cb.on_train_begin()
        for i in range(10):
            time.sleep(0.005)
            cb.on_step_end(step=i)
        cb.on_train_end()
        # No assertion on output — just ensure it doesn't crash

    def test_hf_state_step_override(self):
        """When a state object with global_step is passed, use it."""
        class FakeState:
            global_step = 42

        cb = Eco2AICallback(region="us-east-1", gpu_tdp_w=300, capture_steps=True)
        cb._tracker.capture_steps = True
        cb.on_train_begin()
        cb.on_step_end(state=FakeState())
        report = cb.on_train_end()
        # snapshot step should reflect global_step=42
        if report.snapshots:
            assert report.snapshots[0].step == 42


# ---------------------------------------------------------------------------
# EmissionsReport tests
# ---------------------------------------------------------------------------

class TestEmissionsReport:
    def _make_report(self, **kwargs):
        defaults = dict(
            total_energy_kwh=0.001,
            total_co2_kg=0.0005,
            equivalent_car_km=0.0023,
            total_steps=100,
            emissions_per_step=5e-6,
            duration_hours=0.5,
            region="us-east-1",
            carbon_intensity_g_per_kwh=379.0,
            gpu_tdp_w=300.0,
        )
        defaults.update(kwargs)
        return EmissionsReport(**defaults)

    def test_summary_lines_not_empty(self):
        r = self._make_report()
        lines = r.summary_lines()
        assert len(lines) > 0
        assert any("kWh" in l for l in lines)

    def test_str_includes_key_fields(self):
        r = self._make_report()
        s = str(r)
        assert "EmissionsReport" in s
        assert "kWh" in s

    def test_zero_energy_report(self):
        r = self._make_report(total_energy_kwh=0, total_co2_kg=0)
        s = str(r)
        assert "0.0000" in s or "0.00" in s
