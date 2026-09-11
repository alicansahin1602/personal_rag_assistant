import json
import os
import csv
from pathlib import Path
import asyncio

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric
)

from generation_raw import retrieve_relevant_chunks, build_prompt, GEN_MODEL, client, number_unique_sources

EVAL_SET_PATH = Path("data/eval/eval_dataset.jsonl")
RESULTS_CSV_PATH = Path("data/eval/eval_results.csv")

# Evaluations make many judge calls in a short period. Use a dedicated client
# with extra retries so temporary Groq 429 responses honor their retry delay.
EVAL_CLIENT = client.with_options(max_retries=10)

class JudgeModel(DeepEvalBaseLLM):
    def __init__(self, model_name: str = GEN_MODEL, client=EVAL_CLIENT):
        self.client = client
        self.model_name = model_name
        super().__init__(model_name=model_name)

    def load_model(self):
        # No additional loading needed for the Groq API client
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()

    def get_model_name(self):
        return self.model_name

    async def a_generate(self, prompt: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt)


judge_model = JudgeModel(client=EVAL_CLIENT, model_name=GEN_MODEL)

METRICS = [
    FaithfulnessMetric(model=judge_model, threshold=0.7),
    AnswerRelevancyMetric(model=judge_model, threshold=0.7),
    ContextualPrecisionMetric(model=judge_model, threshold=0.7),
    ContextualRecallMetric(model=judge_model, threshold=0.7),
]

# Keep only one test case/metric active at a time and leave a short gap between
# dispatches. Increase throttle_value if the organization's OTPM limit is still
# reached during a large evaluation.
ASYNC_CONFIG = AsyncConfig(run_async=True, max_concurrent=1, throttle_value=1)

def load_eval_set(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def run_pipeline(question: str) -> tuple[str, list[str]]:

    chunks = retrieve_relevant_chunks(question)
    chunks, source_numbers = number_unique_sources(chunks)
    contexts = [c['text'] for c in chunks]
    prompt = build_prompt(question, chunks)

    response = EVAL_CLIENT.chat.completions.create(
        model=GEN_MODEL,
        messages = [{'role' : 'user', 'content' : prompt}],
        temperature=0.5
    )

    answer = response.choices[0].message.content
    return answer, contexts


def main():
    eval_items = load_eval_set(EVAL_SET_PATH)

    if not eval_items:
        print(f"No eval examples found in {EVAL_SET_PATH} -- add some first.")
        return 

    print(f"Running pipeline on {len(eval_items)} eval questions...")
    test_cases = []

    for item in eval_items:
        answer, contexts = run_pipeline(item['question'])

        test_cases.append(
            LLMTestCase(
                input = item['question'],
                actual_output= answer,
                retrieval_context= contexts,
                expected_output= item['ground_truth']
            )
        )

    print("Scoring with DeepEval (calls the judge model several times per question)...\n")
    results = evaluate(
        test_cases,
        METRICS,
        async_config=ASYNC_CONFIG,
        display_config=DisplayConfig(file_output_dir="data/eval", file_type="html"),
    )
 
    rows = []
    for tr in results.test_results:
        for md in tr.metrics_data or []:
            rows.append(
                {
                    "question": tr.input,
                    "metric": md.name,
                    "score": round(md.score, 3) if md.score is not None else "",
                    "passed": md.success,
                    "reason": md.reason,
                }
            )
 
    RESULTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "metric", "score", "passed", "reason"])
        writer.writeheader()
        writer.writerows(rows)
 
    print(f"\nSaved {len(rows)} metric results to {RESULTS_CSV_PATH}")
    print("(An HTML report was also saved under data/eval/ by DeepEval itself.)\n")
 
    metric_scores: dict[str, list[float]] = {}
    for r in rows:
        if r["score"] != "":
            metric_scores.setdefault(r["metric"], []).append(r["score"])
 
    print("Average scores:")
    for metric, scores in metric_scores.items():
        print(f"  {metric}: {sum(scores) / len(scores):.3f}")

if __name__ == '__main__':
    main()
