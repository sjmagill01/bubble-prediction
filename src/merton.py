"""
merton.py — Merton model: spread computation, vega, and CIV inversion.

The Merton (1974) structural model prices a firm's debt as a put option
on the firm's assets. Given asset volatility sigma_A, leverage L = D/(D+E),
and debt maturity tau:

    Bond value B = N(d2) + N(-d1)/L
    Spread s = -(1/tau) * ln(B)

where d1 = [-ln(L)] / (sigma_A * sqrt(tau)) + 0.5 * sigma_A * sqrt(tau)
      d2 = d1 - sigma_A * sqrt(tau)

CIV inversion: given observed spread, solve for sigma_A via Newton-Raphson.
"""
import numpy as np
from scipy.stats import norm


def merton_spread(sigma_A, L, tau):
    """
    Compute Merton model credit spread.

    Parameters
    ----------
    sigma_A : array — asset volatility (annualized, decimal)
    L : array — leverage D/(D+E), in (0, 1)
    tau : array — debt maturity in years

    Returns
    -------
    spread : array — model-implied credit spread (decimal)
    """
    sigma_A = np.atleast_1d(np.asarray(sigma_A, dtype=float))
    L = np.atleast_1d(np.asarray(L, dtype=float))
    tau = np.atleast_1d(np.asarray(tau, dtype=float))

    sqrt_tau = np.sqrt(tau)
    d1 = (-np.log(L)) / (sigma_A * sqrt_tau) + 0.5 * sigma_A * sqrt_tau
    d2 = d1 - sigma_A * sqrt_tau
    bond_val = norm.cdf(d2) + norm.cdf(-d1) / L
    spread = -(1.0 / tau) * np.log(np.maximum(bond_val, 1e-300))
    return np.maximum(spread, 0.0)


def merton_vega(sigma_A, L, tau):
    """Compute d(spread)/d(sigma_A) for Newton-Raphson."""
    sigma_A = np.atleast_1d(np.asarray(sigma_A, dtype=float))
    L = np.atleast_1d(np.asarray(L, dtype=float))
    tau = np.atleast_1d(np.asarray(tau, dtype=float))

    sqrt_tau = np.sqrt(tau)
    d1 = (-np.log(L)) / (sigma_A * sqrt_tau) + 0.5 * sigma_A * sqrt_tau
    d2 = d1 - sigma_A * sqrt_tau
    dd1 = np.log(L) / (sigma_A ** 2 * sqrt_tau) + 0.5 * sqrt_tau
    dd2 = dd1 - sqrt_tau
    B = np.maximum(norm.cdf(d2) + norm.cdf(-d1) / L, 1e-300)
    dB = norm.pdf(d2) * dd2 - norm.pdf(-d1) * dd1 / L
    return -(1.0 / tau) * dB / B


def invert_spread_to_civ(spread, L, tau, max_iter=50, tol=1e-10):
    """
    Newton-Raphson inversion: observed spread → CIV (asset volatility).

    Parameters
    ----------
    spread : array — observed credit spread (decimal)
    L : array — leverage D/(D+E)
    tau : array — debt maturity in years

    Returns
    -------
    civ : array — credit-implied asset volatility (decimal). NaN on failure.
    """
    spread = np.atleast_1d(np.asarray(spread, dtype=float))
    L = np.atleast_1d(np.asarray(L, dtype=float))
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    n = len(spread)

    sigma = np.full(n, 0.25)  # initial guess
    valid = (spread > 0) & (L > 0) & (L < 1) & (tau > 0)

    for _ in range(max_iter):
        s_model = merton_spread(sigma[valid], L[valid], tau[valid])
        v = merton_vega(sigma[valid], L[valid], tau[valid])
        residual = s_model - spread[valid]
        safe_v = np.where(np.abs(v) > 1e-20, v, 1e-20)
        delta = np.clip(residual / safe_v, -0.5, 0.5)
        sigma[valid] = np.clip(sigma[valid] - delta, 1e-6, 10.0)
        if np.max(np.abs(delta)) < tol:
            break

    # Verify convergence
    s_check = merton_spread(sigma, L, tau)
    bad = np.abs(s_check - spread) > 1e-5
    sigma[bad | ~valid] = np.nan
    return sigma
