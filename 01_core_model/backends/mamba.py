"""Mamba backend adapter.

This fixed BXMNet bottleneck currently targets **mambapy** (as per your working setup).
If you later switch to another implementation, you can change only this file.
"""

from __future__ import annotations

try:
    from mambapy.mamba import Mamba, MambaConfig  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Could not import 'mambapy'. Install it (the same wheel/commit you used on the cluster) "
        "or adjust bxmnet/backends/mamba.py to your preferred Mamba implementation."
    ) from e

__all__ = ["Mamba", "MambaConfig"]
