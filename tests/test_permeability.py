import pytest

from basalt_processing.permeability import calculate_permeability


def test_calculate_permeability_uses_darcy_law_for_a_cylinder():
    result = calculate_permeability(
        flow_rate_m3_s=2e-12,
        viscosity_pa_s=1e-3,
        diameter_m=0.02,
        length_m=0.04,
        pressure_upstream_pa=10e6,
        pressure_downstream_pa=6e6,
    )

    assert result == pytest.approx(6.366197723675813e-20)
