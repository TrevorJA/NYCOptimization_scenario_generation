"""Tests for the namespaced generator seeds (cross-design disjointness contract)."""

from __future__ import annotations

import pytest

import scengen
from scengen.seeds import SEED_DOMAINS, design_seed


DRAWS = range(20)


def test_design_seed_is_deterministic():
    assert design_seed(0, "du_pool", 3) == design_seed(0, "du_pool", 3)
    assert design_seed(7, "etest:kn") == design_seed(7, "etest:kn", 0)  # draw defaults to 0


def test_design_seed_in_valid_rng_range():
    for domain in SEED_DOMAINS:
        for draw in DRAWS:
            s = design_seed(0, domain, draw)
            assert isinstance(s, int)
            assert 0 <= s < 2**31 - 1


def test_design_seed_varies_with_root_domain_and_draw():
    base = design_seed(0, "stat_pool", 0)
    assert design_seed(1, "stat_pool", 0) != base  # root
    assert design_seed(0, "du_pool", 0) != base  # domain
    assert design_seed(0, "stat_pool", 1) != base  # draw


def test_no_collisions_across_canonical_domains_and_draws():
    """The disjointness contract: no two (domain, draw) pairs may share a seed.

    Under the old ``root + draw`` scheme, draw k of every design collided by construction. Now that
    each design GENERATES its own realizations, a collision would make two designs draw correlated
    realizations — the exact confound the control removes.
    """
    seeds = {
        (domain, draw): design_seed(0, domain, draw)
        for domain in sorted(SEED_DOMAINS)
        for draw in DRAWS
    }
    assert len(set(seeds.values())) == len(seeds) == len(SEED_DOMAINS) * len(DRAWS)


@pytest.mark.parametrize("root", [0, 1, 12345])
def test_no_collisions_for_other_roots(root):
    seeds = [design_seed(root, d, k) for d in sorted(SEED_DOMAINS) for k in DRAWS]
    assert len(set(seeds)) == len(seeds)


def test_canonical_domains_are_frozen_and_complete():
    assert isinstance(SEED_DOMAINS, frozenset)
    assert SEED_DOMAINS == {
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


def test_exported_from_package():
    assert scengen.design_seed is design_seed
    assert scengen.SEED_DOMAINS is SEED_DOMAINS
