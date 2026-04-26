import logging
logging.basicConfig(level=logging.INFO)

from app.services.embedding_service import EmbeddingService
from app.core.vector_store import VectorStore
from app.services.retrieval_service import RetrievalService
from app.services.evaluation_service import EvaluationService

# wire up services
emb_service = EmbeddingService()
vs          = VectorStore()
ret_service = RetrievalService(
    embedding_service=emb_service,
    vector_store=vs,
)
eval_service = EvaluationService(retrieval_service=ret_service)

# golden test set — 5 questions with known ground truths
# these are drawn directly from the book content
test_cases = [
    {
        "question": "What is the nearest neighbor heuristic?",
        "ground_truth": (
            "The nearest neighbor heuristic starts at a point and "
            "repeatedly walks to the closest unvisited point until "
            "all points are visited, then returns to the start."
        ),
    },
    {
        "question": "What is the time complexity of BFS?",
        "ground_truth": (
            "BFS runs in O(n + m) time where n is the number of "
            "vertices and m is the number of edges."
        ),
    },
    {
        "question": "What is dynamic programming?",
        "ground_truth": (
            "Dynamic programming solves problems by breaking them "
            "into overlapping subproblems and storing results to "
            "avoid redundant computation."
        ),
    },
    {
        "question": "What is the difference between DFS and BFS?",
        "ground_truth": (
            "BFS explores vertices level by level using a queue. "
            "DFS explores as far as possible along each branch "
            "using a stack or recursion."
        ),
    },
    {
        "question": "What is a heap data structure?",
        "ground_truth": (
            "A heap is a tree-based data structure that satisfies "
            "the heap property: the parent is always greater than "
            "or equal to its children in a max-heap."
        ),
    },
]

print("Running RAGAS evaluation on 5 test cases...")
print("This will take 2-3 minutes...\n")

scores = eval_service.evaluate_batch(test_cases)

print("\n" + "="*50)
print("RAGAS EVALUATION RESULTS")
print("="*50)

if "error" in scores:
    print(f"Error: {scores['error']}")
else:
    print(f"faithfulness      : {scores['faithfulness']}")
    print(f"answer_relevancy  : {scores['answer_relevancy']}")
    print(f"context_precision : {scores['context_precision']}")
    print(f"num_samples       : {scores['num_samples']}")

    print("\nPER QUESTION BREAKDOWN:")
    for s in scores["per_sample_scores"]:
        print(f"\n  Q: {s['question'][:55]}")
        print(f"     faithfulness      : {s['faithfulness']}")
        print(f"     answer_relevancy  : {s['answer_relevancy']}")
        print(f"     context_precision : {s['context_precision']}")