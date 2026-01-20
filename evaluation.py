"""
RAG Pipeline Evaluation Framework

Evaluates retrieval and generation quality using:
- Retrieval metrics: Context Precision, Context Recall
- Generation metrics: Faithfulness, Answer Relevancy

Usage:
    python evaluation.py                    # Run evaluation on test dataset
    python evaluation.py --create-dataset   # Interactive dataset creation
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from generation import generate_answer, load_documents_for_bm25

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Evaluation LLM (used for LLM-as-judge metrics)
EVAL_MODEL = "gpt-4o-mini"
EVAL_TEMPERATURE = 0.0

# Test dataset path
DATASET_PATH = Path(__file__).parent / "eval_dataset.json"


@dataclass
class TestCase:
    """A single evaluation test case."""
    question: str
    ground_truth: str  # Expected answer
    expected_sources: list[str] = None  # Expected source files (for retrieval eval)

    def to_dict(self):
        return asdict(self)


@dataclass
class EvalResult:
    """Results for a single test case."""
    question: str
    ground_truth: str
    generated_answer: str
    retrieved_sources: list[str]
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float

    @property
    def overall_score(self) -> float:
        """Weighted average of all metrics."""
        return (
            self.context_precision * 0.2 +
            self.context_recall * 0.2 +
            self.faithfulness * 0.3 +
            self.answer_relevancy * 0.3
        )


def get_eval_llm():
    """Get LLM for evaluation judging."""
    return ChatOpenAI(
        model=EVAL_MODEL,
        temperature=EVAL_TEMPERATURE,
        openai_api_key=OPENAI_API_KEY
    )


# --- Retrieval Metrics ---

def compute_context_precision(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    """
    Measures what fraction of retrieved documents are relevant.
    Precision = |relevant ∩ retrieved| / |retrieved|
    """
    if not retrieved_sources:
        return 0.0
    if not expected_sources:
        return 1.0  # No expectations = assume all relevant

    retrieved_set = set(retrieved_sources)
    expected_set = set(expected_sources)
    relevant_retrieved = retrieved_set & expected_set

    return len(relevant_retrieved) / len(retrieved_set)


def compute_context_recall(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    """
    Measures what fraction of relevant documents were retrieved.
    Recall = |relevant ∩ retrieved| / |relevant|
    """
    if not expected_sources:
        return 1.0  # No expectations = perfect recall

    retrieved_set = set(retrieved_sources)
    expected_set = set(expected_sources)
    relevant_retrieved = retrieved_set & expected_set

    return len(relevant_retrieved) / len(expected_set)


# --- Generation Metrics (LLM-as-Judge) ---

FAITHFULNESS_PROMPT = """You are evaluating whether an answer is faithful to the provided context.

An answer is faithful if ALL claims in the answer can be directly supported by the context.
An answer is NOT faithful if it contains information not present in the context (hallucination).

Context:
{context}

Answer:
{answer}

Rate the faithfulness on a scale of 0.0 to 1.0:
- 1.0: All claims are directly supported by the context
- 0.5: Some claims are supported, some are not
- 0.0: Most claims are not supported or contradicted

Respond with ONLY a number between 0.0 and 1.0, nothing else."""


RELEVANCY_PROMPT = """You are evaluating whether an answer is relevant to the question asked.

Question: {question}

Answer: {answer}

Ground Truth (expected answer): {ground_truth}

Rate the answer relevancy on a scale of 0.0 to 1.0:
- 1.0: Answer fully addresses the question and matches the expected answer
- 0.7: Answer mostly addresses the question with minor gaps
- 0.5: Answer partially addresses the question
- 0.3: Answer is tangentially related but misses key points
- 0.0: Answer is completely off-topic or wrong

Respond with ONLY a number between 0.0 and 1.0, nothing else."""


def compute_faithfulness(answer: str, context: str) -> float:
    """Use LLM to judge if answer is faithful to context."""
    llm = get_eval_llm()
    prompt = ChatPromptTemplate.from_template(FAITHFULNESS_PROMPT)
    chain = prompt | llm

    try:
        result = chain.invoke({"context": context, "answer": answer})
        score = float(result.content.strip())
        return max(0.0, min(1.0, score))
    except (ValueError, AttributeError):
        return 0.5  # Default on parse failure


def compute_answer_relevancy(question: str, answer: str, ground_truth: str) -> float:
    """Use LLM to judge if answer is relevant to the question."""
    llm = get_eval_llm()
    prompt = ChatPromptTemplate.from_template(RELEVANCY_PROMPT)
    chain = prompt | llm

    try:
        result = chain.invoke({
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth
        })
        score = float(result.content.strip())
        return max(0.0, min(1.0, score))
    except (ValueError, AttributeError):
        return 0.5  # Default on parse failure


# --- Evaluation Runner ---

def evaluate_single(test_case: TestCase, verbose: bool = False) -> EvalResult:
    """Evaluate a single test case."""
    # Generate answer using the RAG pipeline
    result = generate_answer(test_case.question, verbose=True)

    answer = result["answer"]
    sources = [s["file"] for s in result["sources"]]

    # Build context from retrieved documents
    context = "\n\n".join([
        doc["content"] for doc in result.get("retrieved_documents", [])
    ])

    # Compute retrieval metrics
    ctx_precision = compute_context_precision(sources, test_case.expected_sources)
    ctx_recall = compute_context_recall(sources, test_case.expected_sources)

    # Compute generation metrics
    faithfulness = compute_faithfulness(answer, context)
    relevancy = compute_answer_relevancy(test_case.question, answer, test_case.ground_truth)

    if verbose:
        print(f"\n  Question: {test_case.question[:60]}...")
        print(f"  Sources: {sources}")
        print(f"  Context Precision: {ctx_precision:.2f}")
        print(f"  Context Recall: {ctx_recall:.2f}")
        print(f"  Faithfulness: {faithfulness:.2f}")
        print(f"  Answer Relevancy: {relevancy:.2f}")

    return EvalResult(
        question=test_case.question,
        ground_truth=test_case.ground_truth,
        generated_answer=answer,
        retrieved_sources=sources,
        context_precision=ctx_precision,
        context_recall=ctx_recall,
        faithfulness=faithfulness,
        answer_relevancy=relevancy,
    )


def run_evaluation(test_cases: list[TestCase], verbose: bool = True) -> dict:
    """Run evaluation on all test cases and return aggregate metrics."""
    print("=" * 60)
    print("RAG Pipeline Evaluation")
    print("=" * 60)
    print(f"\nEvaluating {len(test_cases)} test cases...")

    results = []
    for i, tc in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] Evaluating...")
        result = evaluate_single(tc, verbose=verbose)
        results.append(result)

    # Compute aggregate metrics
    n = len(results)
    avg_ctx_precision = sum(r.context_precision for r in results) / n
    avg_ctx_recall = sum(r.context_recall for r in results) / n
    avg_faithfulness = sum(r.faithfulness for r in results) / n
    avg_relevancy = sum(r.answer_relevancy for r in results) / n
    avg_overall = sum(r.overall_score for r in results) / n

    report = {
        "num_test_cases": n,
        "metrics": {
            "context_precision": round(avg_ctx_precision, 3),
            "context_recall": round(avg_ctx_recall, 3),
            "faithfulness": round(avg_faithfulness, 3),
            "answer_relevancy": round(avg_relevancy, 3),
            "overall_score": round(avg_overall, 3),
        },
        "results": [
            {
                "question": r.question,
                "context_precision": r.context_precision,
                "context_recall": r.context_recall,
                "faithfulness": r.faithfulness,
                "answer_relevancy": r.answer_relevancy,
                "overall": r.overall_score,
            }
            for r in results
        ]
    }

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nTest Cases: {n}")
    print("\nAggregate Metrics:")
    print(f"  Context Precision:  {avg_ctx_precision:.3f}")
    print(f"  Context Recall:     {avg_ctx_recall:.3f}")
    print(f"  Faithfulness:       {avg_faithfulness:.3f}")
    print(f"  Answer Relevancy:   {avg_relevancy:.3f}")
    print(f"  ─────────────────────────────")
    print(f"  Overall Score:      {avg_overall:.3f}")

    return report


# --- Dataset Management ---

def load_dataset() -> list[TestCase]:
    """Load test dataset from JSON file."""
    if not DATASET_PATH.exists():
        print(f"No dataset found at {DATASET_PATH}")
        print("Run with --create-dataset to create one.")
        return []

    with open(DATASET_PATH, "r") as f:
        data = json.load(f)

    return [
        TestCase(
            question=tc["question"],
            ground_truth=tc["ground_truth"],
            expected_sources=tc.get("expected_sources"),
        )
        for tc in data["test_cases"]
    ]


def save_dataset(test_cases: list[TestCase]):
    """Save test dataset to JSON file."""
    data = {
        "test_cases": [tc.to_dict() for tc in test_cases]
    }
    with open(DATASET_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(test_cases)} test cases to {DATASET_PATH}")


def create_dataset_interactive():
    """Interactive CLI for creating test cases."""
    print("=" * 60)
    print("Create Evaluation Dataset")
    print("=" * 60)

    # Load existing or start fresh
    if DATASET_PATH.exists():
        existing = load_dataset()
        print(f"\nFound {len(existing)} existing test cases.")
        choice = input("Append to existing? (y/n): ").strip().lower()
        test_cases = existing if choice == 'y' else []
    else:
        test_cases = []

    # Show available source files
    print("\nLoading documents to show available sources...")
    docs = load_documents_for_bm25()
    sources = sorted(set(d.metadata.get("source_file", "") for d in docs))
    print(f"\nAvailable sources ({len(sources)}):")
    for s in sources:
        print(f"  - {s}")

    print("\n" + "-" * 40)
    print("Enter test cases (empty question to finish)")
    print("-" * 40)

    while True:
        print()
        question = input("Question: ").strip()
        if not question:
            break

        ground_truth = input("Expected answer: ").strip()
        if not ground_truth:
            print("Skipping - ground truth required")
            continue

        expected_sources_input = input("Expected sources (comma-separated, or empty): ").strip()
        expected_sources = None
        if expected_sources_input:
            expected_sources = [s.strip() for s in expected_sources_input.split(",")]

        test_cases.append(TestCase(
            question=question,
            ground_truth=ground_truth,
            expected_sources=expected_sources,
        ))
        print(f"Added test case #{len(test_cases)}")

    if test_cases:
        save_dataset(test_cases)
    else:
        print("No test cases to save.")


def create_sample_dataset():
    """Create a sample dataset for demonstration."""
    sample_cases = [
        TestCase(
            question="How can I perform barcode scanning on an emulated Android device?",
            ground_truth="You can use the Android emulator's extended controls to simulate barcode scanning, or use a virtual camera with barcode images.",
            expected_sources=["Barcode Scanning.pdf"],
        ),
        TestCase(
            question="What are the coding standards for the frontend?",
            ground_truth="The frontend follows specific architectural styles and coding standards documented in the Frontend Architectural Style guide.",
            expected_sources=["Frontend Architectural Style.pdf"],
        ),
        TestCase(
            question="How should I write tests in the ServiceCore project?",
            ground_truth="Follow the testing practices guide which covers unit tests, integration tests, and test organization.",
            expected_sources=["Testing Practices.pdf"],
        ),
    ]

    save_dataset(sample_cases)
    print("\nCreated sample dataset with 3 test cases.")
    print("Edit eval_dataset.json to add more test cases.")


def main():
    import sys

    if "--create-dataset" in sys.argv:
        create_dataset_interactive()
    elif "--sample-dataset" in sys.argv:
        create_sample_dataset()
    else:
        test_cases = load_dataset()
        if not test_cases:
            print("\nTo get started:")
            print("  python evaluation.py --sample-dataset   # Create sample dataset")
            print("  python evaluation.py --create-dataset   # Create custom dataset")
            return

        report = run_evaluation(test_cases)

        # Save report
        report_path = Path(__file__).parent / "eval_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nDetailed report saved to: {report_path}")


if __name__ == "__main__":
    main()
