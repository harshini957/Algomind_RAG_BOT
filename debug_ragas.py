import logging
logging.basicConfig(level=logging.INFO)

from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from app.services.evaluation_service import GroqSafeChatOpenAI
from app.config import settings

samples = [
    SingleTurnSample(
        user_input="What is BFS?",
        response=(
            "BFS stands for Breadth-First Search. It explores a graph "
            "level by level using a queue. Time complexity is O(n+m)."
        ),
        retrieved_contexts=[
            "Breadth-First Search (BFS) explores vertices level by level "
            "using a queue. It runs in O(n+m) time where n is vertices "
            "and m is edges.",
        ],
        reference="BFS is a graph traversal that uses a queue, O(n+m) time.",
    )
]

eval_dataset = EvaluationDataset(samples=samples)

judge_llm = LangchainLLMWrapper(
    GroqSafeChatOpenAI(
        model="llama-3.3-70b-versatile",
        api_key=settings.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
        temperature=0,
    )
)

judge_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"trust_remote_code": True},
    )
)

print("Running single-sample RAGAS test with n=1 fix...")
scores = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy],
    llm=judge_llm,
    embeddings=judge_embeddings,
)

print(f"\nfaithfulness    : {scores.scores[0].get('faithfulness')}")
print(f"answer_relevancy: {scores.scores[0].get('answer_relevancy')}")