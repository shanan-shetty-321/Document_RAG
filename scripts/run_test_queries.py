"""
Populate the usage log with realistic data.

Fires a batch of test questions at the running FastAPI /ask endpoint - a mix of
genuinely answerable questions (grounded in specific contract sections) and
out-of-scope ones (to exercise the not-found path). A few popular questions are
repeated so the "most frequently asked" analytic has something to show.

Prerequisite: the API must be running (uvicorn app.main:app). Run from the
project root:  python scripts/run_test_queries.py
"""

import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Answerable questions, each maps to a real section of the AWS Customer Agreement.
ANSWERABLE = [
    "What interest rate does AWS charge on late payments?",            # 3.1
    "How much notice does AWS give before discontinuing a material functionality?",  # 1.5
    "How can I terminate the agreement for convenience?",              # 5.2(a)
    "How much advance notice must AWS give to terminate for convenience?",  # 5.2(a)
    "What is the cap on AWS's total liability?",                       # 9.2
    "How long must I keep AWS Confidential Information confidential after the term ends?",  # 11.9
    "What does 'Your Content' mean?",                                  # 12
    "What does 'AWS Content' mean?",                                   # 12
    "Where are disputes resolved if my contracting party is Amazon Web Services, Inc.?",  # gov. law
    "How much notice is required for adverse changes to a Service Level Agreement?",  # 1.6
    "Can I assign the agreement to another company?",                  # 11.1
    "Who is responsible for paying taxes under the agreement?",        # 3.2
    "What happens to my content after the agreement is terminated?",   # 5.3
    "When can AWS suspend my account?",                                # 4.1
    "Are the AWS services provided with any warranties?",              # 8
    "How much notice does AWS give before increasing fees?",           # 3.1
    "Does AWS provide support to End Users?",                          # 3 / 2.5
    "Who owns the intellectual property rights in Suggestions I provide?",  # 6.5
    "In what language must notices under the agreement be given?",     # 11.8
    "Can I create more than one AWS account per email address?",       # 2.1
    "What does 'Acceptable Use Policy' mean?",                         # 12
    "What law governs the agreement for AWS India?",                   # gov. law
    "What are my responsibilities for securing and backing up my content?",  # 2.3
    "How does AWS handle events beyond its control such as natural disasters?",  # 11.3
    "What is the arbitration body for disputes in Australia?",         # 11.5(d)
    "Can AWS modify this agreement, and how?",                         # 10
]

# Out-of-scope questions -> should return "not found" (no hallucination).
OUT_OF_SCOPE = [
    "What is the price of an Amazon EC2 instance?",
    "What is the capital of France?",
    "How do I bake chocolate chip cookies?",
    "What is the maximum timeout for an AWS Lambda function?",
    "Who is the current CEO of Amazon?",
    "What is the weather forecast for tomorrow?",
]

# Repeat a few popular questions so "most frequent" is meaningful.
REPEATS = [
    "What interest rate does AWS charge on late payments?",
    "What interest rate does AWS charge on late payments?",
    "How can I terminate the agreement for convenience?",
    "What does 'Your Content' mean?",
]

ALL_QUESTIONS = ANSWERABLE + OUT_OF_SCOPE + REPEATS


def main() -> None:
    print(f"Sending {len(ALL_QUESTIONS)} test queries to {API_URL}/ask ...\n")
    found = notfound = errors = 0

    for i, question in enumerate(ALL_QUESTIONS, 1):
        try:
            resp = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            flag = "FOUND    " if data["answer_found"] else "NOT FOUND"
            found += data["answer_found"]
            notfound += not data["answer_found"]
            print(f"[{i:>2}/{len(ALL_QUESTIONS)}] {flag} "
                  f"(score={data.get('top_score', 0):.2f}, {data.get('latency_ms', 0):.0f} ms)  {question}")
        except Exception as exc:  # noqa: BLE001 - this is a test harness
            errors += 1
            print(f"[{i:>2}/{len(ALL_QUESTIONS)}] ERROR: {exc}  ({question})")

        # Stay under Groq's free-tier rate limit (~30 requests/min).
        time.sleep(2)

    print(f"\nDone. answered={found}  not_found={notfound}  errors={errors}")
    print(f"View analytics at {API_URL}/analytics")


if __name__ == "__main__":
    main()
