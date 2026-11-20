import pytest

from greeks import analytical, finite_difference

CASES = [
    (100, 100, 0.05, 1.0, 0.2),
    (100, 120, 0.03, 0.5, 0.35),
    (100, 80, 0.01, 0.1, 0.5),
    (2500, 2400, 0.04, 2.0, 0.25),
]
GREEKS = ["delta", "gamma", "vega", "theta", "rho"]


def test_reference_values():
    call = analytical.compute_all(100, 100, 0.05, 1.0, 0.2)["call"]
    assert call["delta"] == pytest.approx(0.6368, abs=1e-4)
    assert call["gamma"] == pytest.approx(0.0188, abs=1e-4)
    assert call["vega"] == pytest.approx(0.3752, abs=1e-4)
    assert call["theta"] == pytest.approx(-0.0176, abs=1e-4)
    assert call["rho"] == pytest.approx(53.232, abs=1e-3)


@pytest.mark.parametrize("s,k,r,t,sigma", CASES)
@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("greek", GREEKS)
def test_analytical_matches_finite_difference(s, k, r, t, sigma, option_type, greek):
    exact = analytical.compute_all(s, k, r, t, sigma)[option_type][greek]
    numeric = finite_difference.compute_all(s, k, r, t, sigma)[option_type][greek]
    assert numeric == pytest.approx(exact, rel=1e-4, abs=1e-6)


@pytest.mark.parametrize("s,k,r,t,sigma", CASES)
def test_call_put_delta_relationship(s, k, r, t, sigma):
    g = analytical.compute_all(s, k, r, t, sigma)
    assert g["call"]["delta"] - g["put"]["delta"] == pytest.approx(1.0)


def test_theta_is_negative_for_atm_call():
    assert analytical.compute_all(100, 100, 0.05, 1.0, 0.2)["call"]["theta"] < 0
    assert finite_difference.compute_all(100, 100, 0.05, 1.0, 0.2)["call"]["theta"] < 0
