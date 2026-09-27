"""The shared input every quality check receives.

The context exists so that twelve checks do not each recompute unique counts,
missing ratios, and quartiles over the same frame. Everything measurable once is
measured once, by the profiler, and handed to the checks.

It stays a value object on purpose: it carries the profile, the configuration,
and what is known about the target, and it has no behaviour of its own beyond
small lookups. Checks that need something else compute it themselves from the
frame rather than growing this object into a god context.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

from aidatasetkit.core.config import KitConfig
from aidatasetkit.core.types import ColumnProfile, DatasetProfile, TargetProfile

__all__ = ["QualityContext"]


@dataclass(frozen=True, slots=True)
class QualityContext:
    """What a quality check may rely on besides the frame itself.

    Attributes:
        profile: Measurements already taken over the frame.
        config: Every threshold the checks compare against. No check defines its
            own constant.
        target: Name of the target column, when one is known.
        target_profile: What the task detector concluded about the target, when a
            target was supplied.
        id_column: Name of an identifier column the caller has already declared,
            so that checks do not report it as a surprise.
    """

    profile: DatasetProfile
    config: KitConfig
    target: Hashable | None = None
    target_profile: TargetProfile | None = None
    id_column: Hashable | None = None

    @property
    def feature_profiles(self) -> tuple[ColumnProfile, ...]:
        """Column profiles excluding the target and the declared identifier."""
        excluded = {name for name in (self.target, self.id_column) if name is not None}
        return tuple(
            profile
            for profile in self.profile.column_profiles
            if profile.name not in excluded
        )

    def is_reserved(self, column: Hashable) -> bool:
        """Whether ``column`` is the target or the declared identifier."""
        return column in {
            name for name in (self.target, self.id_column) if name is not None
        }
