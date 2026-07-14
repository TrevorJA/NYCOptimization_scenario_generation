"""Namespaced generator seeds: one place where the cross-design disjointness contract lives.

Each scenario design now **generates its own realizations** rather than selecting indices from a
shared master ensemble. Deriving seeds as ``root + draw`` (the old scheme) would hand draw ``k`` of
design A and draw ``k`` of design B the *same* seed, so the two designs would draw correlated —
often identical — realizations. That reintroduces exactly the cross-design confound the new
statistical control is meant to remove.

:func:`design_seed` therefore namespaces the seed by a **domain** string: seeds are deterministic in
``(root, domain, draw)`` (replication stays exact) but the streams for different domains are
unrelated. :data:`SEED_DOMAINS` is the canonical, frozen list of domains — add new generation streams
here rather than inventing ad-hoc domain strings at call sites, so the disjointness contract stays
auditable in one place.
"""

from __future__ import annotations

import hashlib
from typing import Final

#: Canonical seed domains — one per independent generation stream in the campaign.
#:
#: ``stat_pool`` / ``du_pool``   candidate pools that are later subsampled (must be i.i.d.-sampled)
#: ``fixed``                     fixed probabilistic records
#: ``resample_pool``             resampled-probabilistic pool (Trindade et al. 2017)
#: ``input_strat``               input-space stratified design
#: ``hazard_select_stat`` /
#: ``hazard_select_du``          hazard-filling *selection* RNG (distinct from the pool RNG)
#: ``etest:kn`` / ``etest:hmm``  held-out test ensembles (two structurally different generators)
#: ``cv_axis``                   the optional CV/variance forcing axis, drawn alongside the mean axis
#:                               (a namespaced stream, never ``root_seed + 1``)
SEED_DOMAINS: Final[frozenset[str]] = frozenset(
    {
        "stat_pool",
        "du_pool",
        "fixed",
        "resample_pool",
        "input_strat",
        "hazard_select_stat",
        "hazard_select_du",
        "etest:kn",
        "etest:hmm",
        "cv_axis",
    }
)


def design_seed(root: int, domain: str, draw: int = 0) -> int:
    """Deterministic, namespaced, collision-free generator seed.

    Args:
        root: Campaign-level root seed (fixes the whole campaign's randomness).
        domain: Seed-domain name; should be one of :data:`SEED_DOMAINS`.
        draw: Replicate (ensemble-draw) index within the domain.

    Returns:
        A seed in ``[0, 2**31 - 1)``, a deterministic function of ``(root, domain, draw)`` and
        effectively independent across distinct ``(domain, draw)`` pairs.
    """
    h = hashlib.blake2b(f"{root}|{domain}|{draw}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") % (2**31 - 1)
