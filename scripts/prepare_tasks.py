#!/usr/bin/env python3
"""Select the BFCL long-context subset, hold out a test split, and add long tasks."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BFCL_DATA = (
    ROOT
    / "third_party"
    / "gorilla"
    / "berkeley-function-call-leaderboard"
    / "bfcl_eval"
    / "data"
)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def select_bfcl_tasks(backends: set[str], require: str) -> list[dict]:
    rows = load_jsonl(BFCL_DATA / "BFCL_v4_multi_turn_long_context.json")
    answers = {
        a["id"]: a
        for a in load_jsonl(
            BFCL_DATA / "possible_answer" / "BFCL_v4_multi_turn_long_context.json"
        )
    }
    selected = []
    for row in rows:
        classes = set(row["involved_classes"])
        if not classes <= backends:
            continue
        if require not in classes:
            continue
        entry = {
            "id": row["id"],
            "source": "bfcl_multi_turn_long_context",
            "question": row["question"],
            "initial_config": row["initial_config"],
            "involved_classes": row["involved_classes"],
            "ground_truth": answers.get(row["id"], {}).get("ground_truth"),
            "success_check": None,
            "needs_early_detail": False,
        }
        selected.append(entry)
    return selected


def make_long_extra_tasks() -> list[dict]:
    """
    At least five tasks that grow to 16k-32k tokens via verbose TradingBot dumps.
    At least two require recalling a detail planted early in the session.
    """
    # Shared long-context initial state borrowed from a BFCL TradingBot seed.
    seed = next(
        r
        for r in load_jsonl(BFCL_DATA / "BFCL_v4_multi_turn_long_context.json")
        if r["involved_classes"] == ["TradingBot"]
    )
    base_cfg = seed["initial_config"]["TradingBot"]
    binding_card = str(base_cfg["account_info"]["binding_card"])
    early_order_id = str(next(iter(base_cfg["orders"].keys())))

    tasks = []

    # 1. Early binding card, then dump histories, then ask for the card.
    tasks.append(
        {
            "id": "long_extra_01_binding_card",
            "source": "project_extra",
            "involved_classes": ["TradingBot"],
            "initial_config": {"TradingBot": base_cfg},
            "needs_early_detail": True,
            "question": [
                [
                    {
                        "role": "user",
                        "content": (
                            "Log in is already done. First call get_account_info, then "
                            "get_transaction_history, then get_order_history. Do not skip "
                            "any of these three calls."
                        ),
                    }
                ],
                [
                    {
                        "role": "user",
                        "content": (
                            "Call get_watchlist and get_available_stocks for Technology, "
                            "then fund the account with 500 dollars."
                        ),
                    }
                ],
                [
                    {
                        "role": "user",
                        "content": (
                            "From the account information you retrieved at the start of "
                            "this session, what is the binding card number? Confirm by "
                            "calling get_account_info again and leave the balance unchanged."
                        ),
                    }
                ],
            ],
            "success_check": {
                "type": "early_detail_recalled",
                "field": "binding_card",
                "needle": binding_card,
            },
            "ground_truth": None,
        }
    )

    # 2. Early order id must survive after long dumps.
    tasks.append(
        {
            "id": "long_extra_02_early_order",
            "source": "project_extra",
            "involved_classes": ["TradingBot"],
            "initial_config": {"TradingBot": base_cfg},
            "needs_early_detail": True,
            "question": [
                [
                    {
                        "role": "user",
                        "content": (
                            f"Look up order details for order id {early_order_id}, then "
                            "call get_transaction_history."
                        ),
                    }
                ],
                [
                    {
                        "role": "user",
                        "content": (
                            "Dump get_order_history and get_watchlist. Then add NVDA to "
                            "the watchlist."
                        ),
                    }
                ],
                [
                    {
                        "role": "user",
                        "content": (
                            "Without looking anything else up first, call get_order_details "
                            f"again for the same early order id {early_order_id}."
                        ),
                    }
                ],
            ],
            "success_check": {
                "type": "early_detail_recalled",
                "field": "order_id",
                "needle": early_order_id,
            },
            "ground_truth": None,
        }
    )

    # 3. Grow context then fund to a known balance.
    start_balance = float(base_cfg["account_info"]["balance"])
    tasks.append(
        {
            "id": "long_extra_03_fund_after_dump",
            "source": "project_extra",
            "involved_classes": ["TradingBot"],
            "initial_config": {"TradingBot": base_cfg},
            "needs_early_detail": False,
            "question": [
                [
                    {
                        "role": "user",
                        "content": (
                            "Call get_transaction_history, get_order_history, and "
                            "get_available_stocks for Technology."
                        ),
                    }
                ],
                [
                    {
                        "role": "user",
                        "content": "Fund my account with exactly 1234.0 dollars.",
                    }
                ],
            ],
            "success_check": {
                "type": "account_balance_equals",
                "value": start_balance + 1234.0,
            },
            "ground_truth": None,
        }
    )

    # 4. Watchlist mutation after verbose reads.
    tasks.append(
        {
            "id": "long_extra_04_watchlist",
            "source": "project_extra",
            "involved_classes": ["TradingBot"],
            "initial_config": {"TradingBot": base_cfg},
            "needs_early_detail": False,
            "question": [
                [
                    {
                        "role": "user",
                        "content": (
                            "Call get_transaction_history and get_watchlist, then add "
                            "MSFT to the watchlist."
                        ),
                    }
                ],
                [
                    {
                        "role": "user",
                        "content": "Call get_watchlist again to confirm MSFT is present.",
                    }
                ],
            ],
            "success_check": {
                "type": "watchlist_contains",
                "symbol": "MSFT",
            },
            "ground_truth": None,
        }
    )

    # 5. Many verbose turns, then fund again (withdraw needs an open market).
    tasks.append(
        {
            "id": "long_extra_05_second_fund",
            "source": "project_extra",
            "involved_classes": ["TradingBot"],
            "initial_config": {"TradingBot": base_cfg},
            "needs_early_detail": False,
            "question": [
                [
                    {
                        "role": "user",
                        "content": (
                            "Call get_transaction_history, then get_order_history, then "
                            "get_account_info."
                        ),
                    }
                ],
                [
                    {
                        "role": "user",
                        "content": (
                            "Call get_available_stocks for Technology and get_watchlist."
                        ),
                    }
                ],
                [
                    {
                        "role": "user",
                        "content": "Fund my account with exactly 100.0 dollars.",
                    }
                ],
            ],
            "success_check": {
                "type": "account_balance_equals",
                "value": start_balance + 100.0,
            },
            "ground_truth": None,
        }
    )

    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default=str(ROOT / "configs" / "tool_subset.json"))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "tasks"))
    parser.add_argument("--test-ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=624)
    args = parser.parse_args()

    with open(args.subset, encoding="utf-8") as f:
        subset = json.load(f)
    backends = set(subset["backends"])

    bfcl = select_bfcl_tasks(backends, require="TradingBot")
    rng = random.Random(args.seed)
    rng.shuffle(bfcl)
    n_test = max(1, int(round(len(bfcl) * args.test_ratio)))
    test_rows = bfcl[:n_test]
    train_rows = bfcl[n_test:]

    long_extra = make_long_extra_tasks()
    # Put two early-detail long tasks in the held-out test split.
    long_test = [t for t in long_extra if t["needs_early_detail"]]
    long_train = [t for t in long_extra if not t["needs_early_detail"]]

    out = Path(args.out_dir)
    write_jsonl(out / "train.jsonl", train_rows + long_train)
    write_jsonl(out / "test.jsonl", test_rows + long_test)
    write_jsonl(out / "long_extra.jsonl", long_extra)
    write_jsonl(out / "all_bfcl_subset.jsonl", bfcl)

    summary = {
        "backends": sorted(backends),
        "n_bfcl": len(bfcl),
        "n_train": len(train_rows) + len(long_train),
        "n_test": len(test_rows) + len(long_test),
        "n_long_extra": len(long_extra),
        "n_early_detail_long": sum(1 for t in long_extra if t["needs_early_detail"]),
        "train_ids": [r["id"] for r in train_rows + long_train],
        "test_ids": [r["id"] for r in test_rows + long_test],
    }
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
