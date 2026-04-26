import logging
from typing import Any, List, Optional

from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_core.outputs import ChatResult
from langchain_core.messages import BaseMessage
from langchain_huggingface import HuggingFaceEmbeddings

from app.services.retrieval_service import RetrievalService
from app.config import settings

logger = logging.getLogger(__name__)


class GroqSafeChatOpenAI(ChatOpenAI):
    """
    ChatOpenAI pointed at Groq's OpenAI-compatible endpoint
    with n=1 forced on every call.

    RAGAS internally requests n=3 completions per call.
    Groq only supports n=1. This wrapper intercepts every
    LLM call and sets n=1 before it reaches Groq's API.
    Without this, all RAGAS scores return NaN.
    """

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        kwargs["n"] = 1
        return super()._generate(messages, stop, run_manager, **kwargs)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        kwargs["n"] = 1
        return await super()._agenerate(messages, stop, run_manager, **kwargs)


class EvaluationService:
    """
    RAGAS 0.4.x evaluation using Groq as judge LLM.

    Metrics:
        faithfulness      — did answer stick to context?
        answer_relevancy  — did answer address the question?
        context_precision — were retrieved chunks useful?
    """

    def __init__(self, retrieval_service: RetrievalService):
        self.retrieval_service = retrieval_service

        self.judge_llm = LangchainLLMWrapper(
            GroqSafeChatOpenAI(
                model="llama-3.3-70b-versatile",
                api_key=settings.groq_api_key,
                base_url="https://api.groq.com/openai/v1",
                temperature=0,
                max_retries=3,
            )
        )

        self.judge_embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={"trust_remote_code": True},
            )
        )

        logger.info("[EvaluationService] RAGAS 0.4.x ready with Groq n=1 fix")

    def evaluate_batch(self, test_cases: list[dict]) -> dict:
        """
        Run RAGAS evaluation on a batch of test cases.

        Each test case:
            question     : str — required
            ground_truth : str — required for context_precision
        """
        if not test_cases:
            return {"error": "No test cases provided", "num_samples": 0}

        logger.info(
            f"[EvaluationService] Evaluating {len(test_cases)} cases..."
        )

        questions, answers, contexts, ground_truths = [], [], [], []

        for i, tc in enumerate(test_cases):
            question     = tc["question"]
            ground_truth = tc.get("ground_truth", "")

            logger.info(
                f"[EvaluationService] "
                f"Case {i+1}/{len(test_cases)}: {question[:55]}"
            )

            try:
                result = self.retrieval_service.query(
                    question=question,
                    top_k=5,
                )
                questions.append(question)
                answers.append(result["answer"])
                ground_truths.append(ground_truth)

                # retrieve actual chunk texts for RAGAS context
                # RAGAS verifies answer claims against this text
                # passing labels instead of text causes faithfulness=0
                dense_vec  = self.retrieval_service.embedding_service.embed_query(question)
                sparse_vec = self.retrieval_service.embedding_service.sparse_embed_query(question)
                raw_chunks = self.retrieval_service.vector_store.search_children(
                    query_dense_vector=dense_vec,
                    query_sparse_vector=sparse_vec,
                    top_k=5,
                )
                reranked = self.retrieval_service.embedding_service.rerank(
                    query=question,
                    candidates=raw_chunks,
                    top_n=5,
                )
                contexts.append([r["text"] for r in reranked])

            except Exception as e:
                logger.error(
                    f"[EvaluationService] Case {i+1} failed: {e}"
                )
                continue

        if not questions:
            return {
                "error":       "All test cases failed during retrieval",
                "num_samples": 0,
            }

        # build RAGAS 0.4.x EvaluationDataset
        samples = [
            SingleTurnSample(
                user_input=q,
                response=a,
                retrieved_contexts=c,
                reference=gt,
            )
            for q, a, c, gt in zip(questions, answers, contexts, ground_truths)
        ]
        eval_dataset = EvaluationDataset(samples=samples)

        logger.info("[EvaluationService] Running RAGAS evaluate()...")

        try:
            scores = evaluate(
                dataset=eval_dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                ],
                llm=self.judge_llm,
                embeddings=self.judge_embeddings,
            )

            # extract per-sample scores
            per_sample = []
            for i, s in enumerate(scores.scores):
                per_sample.append({
                    "question":          questions[i] if i < len(questions) else "",
                    "faithfulness":      round(float(s.get("faithfulness") or 0), 4),
                    "answer_relevancy":  round(float(s.get("answer_relevancy") or 0), 4),
                    "context_precision": round(float(s.get("context_precision") or 0), 4),
                })

            # aggregate: mean over valid (non-nan) per-sample scores
            def mean_score(key: str) -> float:
                import math
                vals = [
                    s[key] for s in per_sample
                    if s[key] and not math.isnan(s[key])
                ]
                return round(sum(vals) / len(vals), 4) if vals else 0.0

            result = {
                "faithfulness":      mean_score("faithfulness"),
                "answer_relevancy":  mean_score("answer_relevancy"),
                "context_precision": mean_score("context_precision"),
                "num_samples":       len(questions),
                "per_sample_scores": per_sample,
            }

            logger.info(f"[EvaluationService] Results: {result}")
            return result

        except Exception as e:
            logger.error(f"[EvaluationService] RAGAS failed: {e}")
            return {
                "error":       str(e),
                "num_samples": len(questions),
            }