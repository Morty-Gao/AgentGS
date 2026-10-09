"""Metric calculations over gold examples and recorded conversations.

Nothing in this package submits a task or calls a model.
"""

from agentgs.evaluation.metrics.report import score_prediction, summarize
from agentgs.evaluation.metrics.tools import SCORING_PROTOCOL

__all__ = ["SCORING_PROTOCOL", "score_prediction", "summarize"]
