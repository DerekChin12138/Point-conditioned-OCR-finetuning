"""Objective sample pools + stage mix recipes.

Pools are generated independently (template-diverse). Stages only sample
from pools according to a YAML recipe — they do not re-render pages.
"""

from point_ocr.pools.spec import POOL_SPECS, PoolSpec, get_pool_spec, list_pool_ids

__all__ = ["POOL_SPECS", "PoolSpec", "get_pool_spec", "list_pool_ids"]
