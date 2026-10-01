#!/usr/bin/env python3
"""LangChain integration sketch.

This demo does not require an LLM API key. It shows the shape of the callback
you would register on an `AgentExecutor` so that every tool invocation is
routed through Convolve. When the proxy returns `abstain`, we raise
`ConvolveAbstained`, which you can catch at the top level to stop the agent.

The pattern is:

    tool -> ConvolveCallback.before_tool -> /v1/sessions/{id}/step
         -> if continue: run the tool -> /observation
         -> if abstain:  raise ConvolveAbstained

Install the extra dependency group first:

    pip install langchain-core httpx
"""
from __future__ import annotations

import httpx


class ConvolveAbstained(RuntimeError):
    """Raised when Convolve instructs the agent to stop."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ConvolveSession:
    """Thin synchronous client for one Convolve session."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self._client = httpx.Client(base_url=base_url, timeout=30.0)
        self._sid: str | None = None
        self._playbook_snippet: str | None = None

    @property
    def session_id(self) -> str:
        if self._sid is None:
            raise RuntimeError("session not started")
        return self._sid

    def start(
        self,
        task: str,
        *,
        max_steps: int = 12,
        budget: float = 0.5,
        threshold: float | None = None,
    ) -> str:
        payload: dict = {"task": task, "max_steps": max_steps, "budget": budget}
        if threshold is not None:
            payload["threshold"] = threshold
        self._sid = self._client.post("/v1/sessions", json=payload).json()["session_id"]
        return self._sid

    def before_tool(
        self,
        thought: str,
        action: str,
        action_input: dict,
        estimated_cost: float = 0.0,
    ) -> None:
        """Call before executing a tool. Raises ConvolveAbstained if we must stop."""
        d = self._client.post(
            f"/v1/sessions/{self.session_id}/step",
            json={
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "estimated_cost": estimated_cost,
            },
        ).json()
        if d["decision"] == "abstain":
            self._playbook_snippet = d.get("abstention_message") or ""
            raise ConvolveAbstained(
                d.get("abstention_message") or "Convolve instructed the agent to stop."
            )

    def after_tool(
        self, observation: str, cost: float = 0.0, is_error: bool | None = None
    ) -> None:
        payload: dict = {"observation": observation, "cost": cost}
        if is_error is not None:
            payload["is_error"] = is_error
        self._client.post(
            f"/v1/sessions/{self.session_id}/observation", json=payload
        )

    def end(self, outcome: str, final_answer: str | None = None) -> None:
        self._client.post(
            f"/v1/sessions/{self.session_id}/end",
            json={"outcome": outcome, "final_answer": final_answer},
        )
        self._client.close()

    @property
    def playbook_snippet(self) -> str | None:
        return self._playbook_snippet


# ---------------------------------------------------------------------------
# Sketch of a LangChain integration
# ---------------------------------------------------------------------------
#
# from langchain.agents import AgentExecutor
# from langchain_core.callbacks import BaseCallbackHandler
#
# class ConvolveCallback(BaseCallbackHandler):
#     def __init__(self, session: ConvolveSession) -> None:
#         self._session = session
#         self._pending: tuple[str, dict] | None = None
#
#     def on_tool_start(self, serialized, input_str, **kwargs):
#         tool = serialized.get("name", "tool")
#         self._pending = (tool, {"input": input_str})
#         self._session.before_tool(
#             thought="(from agent)", action=tool, action_input=self._pending[1]
#         )
#
#     def on_tool_end(self, output, **kwargs):
#         self._session.after_tool(observation=str(output))
#
#     def on_tool_error(self, error, **kwargs):
#         self._session.after_tool(observation=str(error), is_error=True)


def demo_without_langchain() -> None:
    """Runs the client without needing LangChain installed."""
    session = ConvolveSession()
    session.start(
        "Book a flight BOM->DEL on 2026-08-15 under Rs.500.", max_steps=10, budget=0.5
    )
    print(f"session: {session.session_id}")

    try:
        for i in range(10):
            session.before_tool(
                thought=f"Attempt {i + 1}",
                action="search_flights",
                action_input={"origin": "BOM", "destination": "DEL", "max_price": 500},
                estimated_cost=0.05,
            )
            session.after_tool(
                observation="No flights under Rs.500. Cheapest is Rs.4,500.",
                cost=0.05,
                is_error=True,
            )
    except ConvolveAbstained as stop:
        print()
        print("Agent stopped by Convolve:")
        print(stop.message)
        session.end("abstained", final_answer=stop.message)


if __name__ == "__main__":
    demo_without_langchain()
