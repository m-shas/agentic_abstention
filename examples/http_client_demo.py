#!/usr/bin/env python3
"""End-to-end demo using the raw HTTP API.

Run the proxy first:

    convolve serve

Then:

    python examples/http_client_demo.py

This script plays the role of both the agent and the tool executor. It walks
a task that is intentionally impossible (a flight under Rs.500) and shows how
Convolve abstains before the agent wastes its full step budget.
"""
from __future__ import annotations

import sys

import httpx

BASE = "http://localhost:8000"


def fake_tool(name: str, payload: dict) -> tuple[str, float, bool]:
    """Stand-in for a real tool. Always returns the same failure for our demo."""
    if name == "search_flights":
        return (
            "No flights found matching criteria. Cheapest available direct "
            "flight is Rs.4,500.",
            0.05,
            True,
        )
    if name == "select_flight":
        return ("Flight not available for the requested criteria.", 0.02, True)
    return (f"Unknown tool '{name}'.", 0.0, True)


def main() -> int:
    task = "Book a flight from BOM to DEL on 2026-08-15 for under Rs.500."

    with httpx.Client(base_url=BASE, timeout=30.0) as client:
        health = client.get("/healthz").json()
        print(f"healthz: {health}")

        session = client.post(
            "/v1/sessions",
            json={"task": task, "max_steps": 10, "budget": 0.50},
        ).json()
        sid = session["session_id"]
        print(f"session: {sid}")
        print(f"task   : {task}")
        print()

        while True:
            proposal = {
                "thought": "Search for the cheapest direct flight.",
                "action": "search_flights",
                "action_input": {
                    "origin": "BOM",
                    "destination": "DEL",
                    "date": "2026-08-15",
                    "max_price": 500,
                },
                "estimated_cost": 0.05,
            }

            d = client.post(f"/v1/sessions/{sid}/step", json=proposal).json()
            print(
                f"step {d['step_index']:>2}  "
                f"P(stop)={d['stop_probability']:.3f}  "
                f"thr={d['threshold']:.2f}  "
                f"decision={d['decision']:>8}  "
                f"reason={d['reason']}"
            )

            if d["decision"] == "abstain":
                print()
                print("ABSTENTION MESSAGE:")
                print(d["abstention_message"])
                break

            observation, cost, is_error = fake_tool(
                proposal["action"], proposal["action_input"]
            )
            client.post(
                f"/v1/sessions/{sid}/observation",
                json={"observation": observation, "cost": cost, "is_error": is_error},
            )

        client.post(
            f"/v1/sessions/{sid}/end",
            json={"outcome": "abstained", "final_answer": "Cheapest is Rs.4,500."},
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
