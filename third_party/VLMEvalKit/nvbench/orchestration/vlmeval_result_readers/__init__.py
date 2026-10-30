"""Strict benchmark-native readers for VLMEvalKit score artifacts."""

from .ocrbench_score_json import OCRBenchScoreJSONError, read_ocrbench_score_json
from .screenspot_score_json import ScreenSpotScoreJSONError, read_screenspot_score_json
from .vlmeval_accuracy_csv import VLMEvalAccuracyCSVError, read_vlmeval_accuracy_csv
from .vlmeval_primary_metric_csv import (
    VLMEvalPrimaryMetricCSVError,
    read_vlmeval_primary_metric_csv,
)

__all__ = (
    "OCRBenchScoreJSONError",
    "ScreenSpotScoreJSONError",
    "VLMEvalAccuracyCSVError",
    "VLMEvalPrimaryMetricCSVError",
    "read_ocrbench_score_json",
    "read_screenspot_score_json",
    "read_vlmeval_accuracy_csv",
    "read_vlmeval_primary_metric_csv",
)
