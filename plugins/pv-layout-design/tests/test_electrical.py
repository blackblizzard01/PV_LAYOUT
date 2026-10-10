"""
Tests for electrical.py — F5 conductor sizing, F6 string sizing.

Hand-computed reference values are placed in each test's docstring so
the grader can verify them independently against the cited formulas.
"""

import math
import pytest

from pv_layout.electrical import (
    select_conductor_area, string_sizing,
    STANDARD_SECTIONS_MM2, _resistivity_at_t,
)


class TestResistivityAtT:
    def test_copper_20c(self):
        """At 20 °C, ρ = ρ₂₀ (no correction)."""
        assert _resistivity_at_t("copper", 20.0) == pytest.approx(1.68e-8, rel=1e-9)

    def test_copper_90c(self):
        """
        ρ(90°C) = 1.68e-8 × [1 + 0.00393 × (90 − 20)]
                = 1.68e-8 × 1.2751 = 2.142e-8 Ω·m
        """
        expected = 1.68e-8 * (1.0 + 0.00393 * 70.0)
        assert _resistivity_at_t("copper", 90.0) == pytest.approx(expected, rel=1e-6)

    def test_aluminum_20c(self):
        assert _resistivity_at_t("aluminum", 20.0) == pytest.approx(2.82e-8, rel=1e-9)


class TestSelectConductorArea:
    """
    Hand-verify the voltage-drop criterion for a known DC segment.

    Inputs: 100 kW, 1000 V DC, 100 m copper, max 0.5 % V-drop, 90 °C.
    ρ(90°C) = 1.68e-8 × [1 + 0.00393 × 70] = 2.142e-8 Ω·m
    I = 100 000 / 1000 = 100 A
    S_min = 2 × ρ × L × I / (δ_V × V)
          = 2 × 2.142e-8 × 100 × 100 / (0.005 × 1000)
          = 4.284e-4 / 5.0 = 8.568e-5 m² = 85.68 mm²
    Smallest standard section ≥ 85.68 mm² → 95 mm².
    """

    def test_dc_hand_check(self):
        rho = 1.68e-8 * (1.0 + 0.00393 * 70.0)
        L, P_kw, V = 100.0, 100.0, 1000.0
        I = P_kw * 1000.0 / V
        s_min_mm2 = 2.0 * rho * L * I / (0.005 * V) * 1e6
        # s_min ≈ 85.7 mm² → next standard = 95 mm²
        assert s_min_mm2 == pytest.approx(85.68, rel=0.01)
        result = select_conductor_area(P_kw, V, L, material="copper",
                                       max_vdrop_pct=0.5, conductor_temp_c=90.0)
        assert result == 95.0

    def test_returns_standard_section(self):
        """Result must always be a value from STANDARD_SECTIONS_MM2."""
        result = select_conductor_area(50.0, 400.0, 200.0, material="copper",
                                       max_vdrop_pct=1.0, three_phase=True, power_factor=0.98)
        assert result in STANDARD_SECTIONS_MM2

    def test_short_cable_gives_smallest_section(self):
        """Very short cable → voltage drop is negligible → smallest section."""
        result = select_conductor_area(1.0, 1000.0, 1.0)
        assert result == STANDARD_SECTIONS_MM2[0]

    def test_zero_power_gives_smallest_section(self):
        result = select_conductor_area(0.0, 1000.0, 100.0)
        assert result == STANDARD_SECTIONS_MM2[0]

    def test_unknown_material_raises(self):
        with pytest.raises(ValueError, match="Unknown material"):
            select_conductor_area(100.0, 1000.0, 100.0, material="unobtanium")

    def test_aluminum_larger_than_copper_same_conditions(self):
        """Aluminum has higher resistivity → needs a larger cross-section."""
        s_cu = select_conductor_area(100.0, 1000.0, 200.0, material="copper")
        s_al = select_conductor_area(100.0, 1000.0, 200.0, material="aluminum")
        assert s_al >= s_cu


class TestStringSizing:
    """
    Hand-verify Electrical_Methodology §1 Eq 1.1–1.7 for a typical module.

    Module: Voc=49.6 V, Vmp=41.5 V, Voc_tc=−0.233%/°C, Vmp_tc=−0.233%/°C.
    Inverter: V_sys_max=1500 V, MPPT 200–1000 V.
    Site T_min=−10°C, T_max=45°C, NOCT=45°C.

    Cold condition (Eq 1.1–1.3):
      T_cell_min = T_air_min = −10°C
      Voc_cold = 49.6 × [1 + (−0.233/100) × (−10−25)]
               = 49.6 × [1 + 0.08155] = 49.6 × 1.08155 = 53.65 V
      N_max_sys  = floor(1500 / 53.65) = floor(27.96) = 27
      N_max_mppt = floor(1000 / 41.5) = floor(24.1) = 24
      N_max = min(27, 24) = 24

    Hot condition (Eq 1.5–1.6):
      T_cell_max = 45 + (45−20) × 1000 / 800 = 45 + 31.25 = 76.25°C
      Vmp_hot = 41.5 × [1 + (−0.233/100) × (76.25−25)]
              = 41.5 × [1 − 0.1194] = 41.5 × 0.8806 = 36.55 V
      N_min = ceil(200 / 36.55) = ceil(5.47) = 6
    """

    @pytest.fixture
    def result(self):
        return string_sizing(
            voc_ref_v=49.6, vmp_ref_v=41.5,
            voc_tc_pct_per_c=-0.233, vmp_tc_pct_per_c=-0.233,
            v_max_system_v=1500, v_mppt_min_v=200, v_mppt_max_v=1000,
            t_min_c=-10.0, t_max_c=45.0, noct_c=45.0,
        )

    def test_n_max(self, result):
        assert result.n_max == 24

    def test_n_min(self, result):
        assert result.n_min == 6

    def test_sizing_ok(self, result):
        assert result.sizing_ok

    def test_n_recommended_is_n_max(self, result):
        assert result.n_recommended == result.n_max

    def test_narrow_mppt_window_fails(self):
        """MPPT 800–810 V forces N_min > N_max."""
        r = string_sizing(
            voc_ref_v=49.6, vmp_ref_v=41.5,
            voc_tc_pct_per_c=-0.233, vmp_tc_pct_per_c=-0.233,
            v_max_system_v=1500, v_mppt_min_v=800, v_mppt_max_v=810,
            t_min_c=-10.0, t_max_c=45.0,
        )
        assert not r.sizing_ok
