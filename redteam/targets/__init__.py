"""Target systems — the things we attack.

A *target* is a system-under-test with a common ``.run(payload) -> RunResult``
interface. It pairs a framing (raw chat vs. document summarisation) with a model
provider, so "the same attack against a local model vs. a frontier model" and
"the same attack sent directly vs. hidden in a document" are both just different
targets in the same matrix.
"""

from .base import TargetSystem
from .direct import DirectTarget
from .document_pipeline import DocumentSummarizationTarget

__all__ = ["TargetSystem", "DirectTarget", "DocumentSummarizationTarget"]
