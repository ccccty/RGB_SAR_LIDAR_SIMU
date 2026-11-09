"""Detection theory for the red/blue evaluation loop (pure numpy, no scipy).

The radar front end thresholds a single sample at ``VT = -ln(Pfa)`` -- correct
for one pulse, but it hides the two numbers an operator actually needs:

* how much SNR a non-fluctuating target needs to reach a given ``P_d``
  (Albersheim's engineering inverse, handbook 3.2);
* what CA-CFAR costs relative to a known noise floor (handbook 3.3), which is
  the price of not knowing the noise level a priori.

Everything here is a closed form or a Poisson sum, so it adds no dependency.
The constants in ``docs/雷达与激光雷达仿真经验公式手册.md`` are used as test
oracles (11.40 dB, 13.11 dB, 1.3 dB, 4.3 dB), not invented numbers.
"""
from __future__ import annotations

import math

__all__ = [
    "threshold_db", "albersheim_snr_db", "ca_cfar_alpha", "ca_cfar_alpha_db",
    "cfar_loss_db", "pd_noncoherent", "marcum_q",
]


def _clip_pfa(pfa):
    return min(max(float(pfa), 1e-12), 0.5)


def threshold_db(pfa):
    """Single-sample square-law threshold in dB: VT = -ln(Pfa).

    Must agree with radar.py's analytic detector (1e-6 -> 11.40 dB).
    """
    return 10.0 * math.log10(max(-math.log(_clip_pfa(pfa)), 1e-12))


def albersheim_snr_db(pfa, pd, n=1):
    """Albersheim's closed-form SNR requirement (handbook 3.2).

    Valid 1e-7 <= Pfa <= 1e-3, 0.1 <= Pd <= 0.9, 1 <= N <= 8096, error < 0.2 dB.
    Oracle: (1e-6, 0.9, 1) -> 13.11 dB (exact 13.18 dB, Albersheim 1981).
    """
    pf, q = _clip_pfa(pfa), float(pd)
    if not (0.0 < q < 1.0):
        raise ValueError("pd must be in (0, 1)")
    pulses = max(int(n), 1)
    a = math.log(0.62 / pf)
    b = math.log(q / (1.0 - q))
    inner = a + 0.12 * a * b + 1.7 * b
    if inner <= 0.0:
        raise ValueError("Albersheim outside its valid range")
    return (-5.0 * math.log10(pulses)
            + (6.2 + 4.54 / math.sqrt(pulses + 0.44)) * math.log10(inner))


def ca_cfar_alpha(nr, pfa):
    """CA-CFAR multiplier alpha = Nr*(Pfa^(-1/Nr) - 1), linear."""
    nr = int(nr)
    if nr < 1:
        raise ValueError("nr must be >= 1")
    return nr * (_clip_pfa(pfa) ** (-1.0 / nr) - 1.0)


def ca_cfar_alpha_db(nr, pfa):
    return 10.0 * math.log10(ca_cfar_alpha(nr, pfa))


def cfar_loss_db(nr, pfa):
    """Excess SNR over a known-noise threshold. Nr=24 @1e-6 -> 1.3 dB."""
    return ca_cfar_alpha_db(nr, pfa) - threshold_db(pfa)


def _gammq(a, x):
    """Regularised upper incomplete gamma Q(a, x).

    Series for x < a+1, continued fraction otherwise (Numerical Recipes).
    """
    if a <= 0.0 or x < 0.0:
        raise ValueError("gammq domain")
    if x == 0.0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1.0:
        ap, term, total = a, 1.0 / a, 1.0 / a
        for _ in range(1000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, 1.0 - total * math.exp(-x + a * math.log(x) - gln))
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return min(max(math.exp(-x + a * math.log(x) - gln) * h, 0.0), 1.0)


def marcum_q(m, a, b):
    """Marcum Q_M(a, b) via the Poisson form of the non-central chi-square."""
    m = max(int(m), 1)
    lam = 0.5 * float(a) ** 2
    vt = 0.5 * float(b) ** 2
    total = 0.0
    pmf = math.exp(-lam)
    limit = int(lam + 12.0 * math.sqrt(lam + 1.0)) + 60
    for k in range(limit):
        if k:
            pmf *= lam / k
        total += pmf * _gammq(m + k, vt)
        if k > lam and pmf < 1e-17:
            break
    return min(max(total, 0.0), 1.0)


def pd_noncoherent(n, snr, vt=None, pfa=None):
    """P_d for a NON-fluctuating target, N pulses, linear SNR per pulse.

    P_d = sum_k Poisson(k; N*S) * Q_gamma(N+k, VT)  (handbook 3.2).
    Pass ``vt`` directly, or ``pfa`` to use the single-sample threshold.
    Oracle: N=1, S=13.18 dB, Pfa=1e-6 -> ~0.90.
    """
    if vt is None:
        if pfa is None:
            raise ValueError("give vt or pfa")
        vt = -math.log(_clip_pfa(pfa))
    return marcum_q(n, math.sqrt(2.0 * max(float(snr), 0.0) * max(int(n), 1)),
                    math.sqrt(2.0 * vt))
