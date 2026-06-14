"""
Serialize and restore auto-tuned scalpel configurations.

A scalpel run is fully determined by the dispersion plugin, the
auto-tuner's selected Bromwich parameters, the grid, and the precision.
This module dumps that configuration to JSON so a later run reproduces
it bit-for-bit (modulo backend nondeterminism, which is documented
separately).

Example
-------
>>> from scalpel.config import save_run_config, load_run_config
>>> save_run_config('run.json',
...     dispersion='maxwell', dispersion_params={'sigma': 1e-3, 'epsilon_r': 4.0},
...     auto_tuner_choices={'a': 1.0, 'T': 20e-9, 'N': 2048},
...     grid={'Nx': 128, 'Ny': 128, 'dx': 0.1},
...     precision='float64', backend='jax_gpu',
... )
>>> cfg = load_run_config('run.json')
>>> cfg.dispersion
'maxwell'
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from typing import Optional


SCHEMA_VERSION = "1.0"


@dataclass
class RunConfig:
    """A fully-specified scalpel run configuration."""

    dispersion: str
    dispersion_params: dict
    auto_tuner_choices: dict
    grid: dict
    precision: str
    backend: str
    schema_version: str = SCHEMA_VERSION
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def save_run_config(
    path: str,
    *,
    dispersion: str,
    dispersion_params: dict,
    auto_tuner_choices: dict,
    grid: dict,
    precision: str,
    backend: str,
    extra: Optional[dict] = None,
) -> None:
    """Write a RunConfig to ``path`` (JSON)."""
    cfg = RunConfig(
        dispersion=dispersion,
        dispersion_params=dict(dispersion_params),
        auto_tuner_choices=dict(auto_tuner_choices),
        grid=dict(grid),
        precision=precision,
        backend=backend,
        extra=dict(extra) if extra else {},
    )
    with open(path, "w") as f:
        json.dump(cfg.as_dict(), f, indent=2, sort_keys=True)


def load_run_config(path: str) -> RunConfig:
    """Read a RunConfig from ``path`` (JSON)."""
    with open(path, "r") as f:
        raw = json.load(f)
    schema = raw.get("schema_version", "0")
    if schema != SCHEMA_VERSION:
        raise ValueError(
            f"config file schema {schema!r} != supported {SCHEMA_VERSION!r}; "
            f"upgrade or downgrade the scalpel version."
        )
    return RunConfig(
        dispersion=raw["dispersion"],
        dispersion_params=raw["dispersion_params"],
        auto_tuner_choices=raw["auto_tuner_choices"],
        grid=raw["grid"],
        precision=raw["precision"],
        backend=raw["backend"],
        schema_version=raw["schema_version"],
        extra=raw.get("extra", {}),
    )


def diff_configs(a: RunConfig, b: RunConfig) -> dict:
    """Field-by-field difference between two configs; useful in tests."""
    diffs = {}
    for k in ("dispersion", "precision", "backend"):
        if getattr(a, k) != getattr(b, k):
            diffs[k] = (getattr(a, k), getattr(b, k))
    for grp in ("dispersion_params", "auto_tuner_choices", "grid", "extra"):
        av = getattr(a, grp)
        bv = getattr(b, grp)
        gd = {}
        for k in set(av) | set(bv):
            if av.get(k) != bv.get(k):
                gd[k] = (av.get(k), bv.get(k))
        if gd:
            diffs[grp] = gd
    return diffs
