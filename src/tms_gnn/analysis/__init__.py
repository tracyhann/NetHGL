"""Repeated-measures statistical models and multiplicity control."""

from tms_gnn.analysis.multiple_testing import (
    benjamini_hochberg,
    holm_adjust,
    max_stat_fwer,
    permute_participant_blocks,
)
from tms_gnn.analysis.preprocessing import ScaledPredictor, scale_by_standard_deviation

__all__ = [
    "benjamini_hochberg",
    "holm_adjust",
    "max_stat_fwer",
    "permute_participant_blocks",
    "ScaledPredictor",
    "scale_by_standard_deviation",
]
