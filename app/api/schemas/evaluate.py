from pydantic import BaseModel
from typing import List


class TestCase(BaseModel):
    question:     str
    ground_truth: str = ""


class EvalRequest(BaseModel):
    test_cases: List[TestCase]


class PerSampleScore(BaseModel):
    question:          str
    faithfulness:      float
    answer_relevancy:  float
    context_precision: float


class EvalResponse(BaseModel):
    faithfulness:      float
    answer_relevancy:  float
    context_precision: float
    num_samples:       int
    per_sample_scores: List[PerSampleScore]