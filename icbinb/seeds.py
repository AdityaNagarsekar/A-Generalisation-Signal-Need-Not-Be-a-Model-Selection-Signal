"""Deterministic seed derivation (plan §5). One outer_run -> a fixed seed quintuple."""
from __future__ import annotations
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class Seeds:
    split_seed: int
    hp_seed: int
    init_seed: int
    loader_seed: int
    proxy_seed: int
    sampler_seed: int
    def asdict(self) -> dict: return asdict(self)

def seeds_for(outer_run: int) -> Seeds:
    b = 10000 * outer_run
    return Seeds(b + 1, b + 2, b + 3, b + 4, b + 5, b + 6)

def init_seed_for(outer_run: int, candidate_id: int) -> int:
    return seeds_for(outer_run).init_seed + candidate_id
