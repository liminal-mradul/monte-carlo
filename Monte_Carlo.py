# Adaptive Monte Carlo π with recursive variance-based interval partitioning
# Quasi–Monte Carlo sampling (van der Corput), mpmath precision.

import time
from dataclasses import dataclass, field
from typing import List, Tuple
import mpmath as mp

# -----------------------------
# Utilities
# -----------------------------

def vdc(n: int, base: int = 2) -> float:
    """
    Van der Corput sequence (base-b radical inverse), in [0, 1).
    Deterministic low-discrepancy points for quasi–Monte Carlo.
    """
    v, denom = 0.0, 1.0
    while n:
        n, rem = divmod(n, base)
        denom *= base
        v += rem / denom
    return v

def f_quarter_circle(x: mp.mpf) -> mp.mpf:
    """
    Integrand for quarter-circle y = sqrt(1 - x^2), x in [0, 1].
    π = 4 * ∫_0^1 sqrt(1 - x^2) dx
    """
    return mp.sqrt(1 - x * x)

@dataclass
class Stratum:
    a: mp.mpf
    b: mp.mpf
    n: int = 0
    sum_fx: mp.mpf = field(default_factory=lambda: mp.mpf('0'))
    sum_fx2: mp.mpf = field(default_factory=lambda: mp.mpf('0'))
    seq_offset: int = 0  # offset into vdc sequence (kept per stratum)

    def width(self) -> mp.mpf:
        return self.b - self.a

    def mean(self) -> mp.mpf:
        return (self.sum_fx / self.n) if self.n > 0 else mp.mpf('0')

    def variance(self) -> mp.mpf:
        if self.n <= 1:
            return mp.mpf('0')
        mu = self.sum_fx / self.n
        # Unbiased sample variance
        return (self.sum_fx2 - self.n * mu * mu) / (self.n - 1)

    def contribution(self) -> mp.mpf:
        # Estimated integral contribution from this stratum
        return self.mean() * self.width()

    def est_variance_contrib(self) -> mp.mpf:
        """
        Estimated variance of the integral contribution from this stratum.
        Var(mean) = Var(X)/n, then scale by width^2 for the integral over [a,b].
        If unsampled, return width^2 to prioritize splitting/sampling.
        """
        if self.n == 0:
            return self.width() ** 2
        return (self.variance() / self.n) * (self.width() ** 2)

def sample_stratum(stratum: Stratum, batch: int, seq_base: int = 2) -> None:
    """
    Quasi–MC sample within [a,b] using van der Corput points with a per-stratum offset.
    """
    a, b = stratum.a, stratum.b
    w = b - a
    start = stratum.seq_offset + 1  # avoid n=0 => u=0 exactly
    for i in range(batch):
        u = vdc(start + i, base=seq_base)  # u in (0,1)
        x = a + w * mp.mpf(u)
        y = f_quarter_circle(x)
        stratum.sum_fx += y
        stratum.sum_fx2 += y * y
    stratum.n += batch
    stratum.seq_offset += batch

# -----------------------------
# Main adaptive estimator
# -----------------------------

def adaptive_pi(
    digits: int = 100,         # working precision (decimal digits)
    max_strata: int = 64,      # maximum number of recursive intervals
    batch_per_iter: int = 256, # samples per stratum per iteration
    max_iters: int = 200,      # safety cap on iterations
    confidence_z: float = 3.0, # ~99.7% CL for error bar
    verbose: bool = False
) -> Tuple[mp.mpf, dict]:
    """
    Adaptive stratified quasi–Monte Carlo for π via 1D integral.
    Returns (pi_estimate, stats_dict).
    """
    mp.mp.dps = max(digits + 10, 50)  # headroom over requested digits
    strata: List[Stratum] = [Stratum(mp.mpf('0'), mp.mpf('1'))]

    total_samples = 0
    t0 = time.time()

    for it in range(max_iters):
        # 1) Sample each stratum once to keep things balanced
        for s in strata:
            sample_stratum(s, batch_per_iter)
            total_samples += batch_per_iter

        # 2) Aggregate estimate and variance
        integral_est = mp.mpf('0')
        var_est = mp.mpf('0')
        for s in strata:
            integral_est += s.contribution()
            var_est += s.est_variance_contrib()

        pi_est = 4 * integral_est
        std_err = mp.sqrt(var_est) * confidence_z * 4
        abs_err = std_err
        denom = mp.fabs(pi_est) or mp.mpf(1)
        rel_err = abs_err / denom

        if verbose:
            print(
                f"iter={it+1:03d} strata={len(strata):03d} samples={total_samples:,} "
                f"pi≈{pi_est}  abs_err≈{abs_err}  rel_err≈{rel_err}"
            )

        # 3) Practical stopping rule for MC (cap at ~30 reliable digits via MC)
        if rel_err < mp.mpf('1e-{}'.format(min(digits, 30))):
            break

        # 4) Split the highest-variance stratum, if allowed
        if len(strata) < max_strata:
            idx = max(range(len(strata)), key=lambda i: strata[i].est_variance_contrib())
            s = strata[idx]
            mid = (s.a + s.b) / 2
            # children inherit offset with a large skip for decorrelation
            left = Stratum(s.a, mid, seq_offset=s.seq_offset)
            right = Stratum(mid, s.b, seq_offset=s.seq_offset + 123457)
            strata.pop(idx)
            strata.extend([left, right])

    t1 = time.time()
    stats = {
        "digits": digits,
        "mpmath_dps": mp.mp.dps,
        "strata": len(strata),
        "total_samples": total_samples,
        "time_sec": t1 - t0,
        "abs_err_est": abs_err,
        "rel_err_est": rel_err,
        "pi_est": pi_est,
    }
    return pi_est, stats

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Fast demo settings (tweak as you like)
    mp.mp.dps = 80  # working precision for arithmetic (can raise)
    pi_est, stats = adaptive_pi(
        digits=80,
        max_strata=32,
        batch_per_iter=512,
        max_iters=20,
        verbose=False
    )
    true_pi = mp.pi
    print("π estimate:", pi_est)
    print("Abs error :", mp.fabs(pi_est - true_pi))
    print("Rel error :", mp.fabs(pi_est - true_pi) / true_pi)
    print("Stats     :", stats)
