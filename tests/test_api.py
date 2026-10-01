from __future__ import annotations

from fastapi.testclient import TestClient


def _start(client: TestClient, task: str = "book a flight under Rs.500") -> str:
    r = client.post(
        "/v1/sessions",
        json={"task": task, "max_steps": 6, "budget": 0.5},
    )
    assert r.status_code == 201, r.text
    return r.json()["session_id"]


def test_healthz(client: TestClient):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert body["state_backend"] in ("memory", "redis")
    assert body["evaluator_source"] in ("heuristic", "trained")


def test_create_and_get_session(client: TestClient):
    sid = _start(client)
    r = client.get(f"/v1/sessions/{sid}")
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == sid
    assert body["status"] == "open"
    assert body["steps"] == []


def test_get_missing_session_returns_404(client: TestClient):
    r = client.get("/v1/sessions/does-not-exist")
    assert r.status_code == 404


def test_step_continue_then_observation(client: TestClient):
    sid = _start(client)
    r = client.post(
        f"/v1/sessions/{sid}/step",
        json={
            "thought": "Search",
            "action": "search_flights",
            "action_input": {"origin": "BOM", "dest": "DEL"},
            "estimated_cost": 0.05,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "continue"
    assert 0.0 <= body["stop_probability"] <= 1.0
    assert body["threshold"] > 0
    assert body["abstention_message"] is None

    r = client.post(
        f"/v1/sessions/{sid}/observation",
        json={"observation": "found 3 flights", "cost": 0.05, "is_error": False},
    )
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert r.json()["step_index"] == 0


def test_observation_without_step_returns_409(client: TestClient):
    sid = _start(client)
    r = client.post(
        f"/v1/sessions/{sid}/observation",
        json={"observation": "x", "cost": 0.0},
    )
    assert r.status_code == 409


def test_double_step_without_observation_returns_409(client: TestClient):
    sid = _start(client)
    client.post(
        f"/v1/sessions/{sid}/step",
        json={"thought": "", "action": "a", "action_input": {}, "estimated_cost": 0.0},
    )
    r = client.post(
        f"/v1/sessions/{sid}/step",
        json={"thought": "", "action": "b", "action_input": {}, "estimated_cost": 0.0},
    )
    assert r.status_code == 409


def test_abstain_when_max_steps_reached(client: TestClient):
    sid = _start(client)
    for i in range(6):
        r = client.post(
            f"/v1/sessions/{sid}/step",
            json={
                "thought": f"attempt {i}",
                "action": "search_flights",
                "action_input": {"i": i},
                "estimated_cost": 0.05,
            },
        )
        assert r.status_code == 200
        if r.json()["decision"] == "abstain":
            assert r.json()["reason"] in ("max_steps_reached", "uncertainty_threshold")
            return
        client.post(
            f"/v1/sessions/{sid}/observation",
            json={"observation": "no results found", "cost": 0.05, "is_error": True},
        )

    # If we never abstained, force a final step to reach the limit.
    r = client.post(
        f"/v1/sessions/{sid}/step",
        json={"thought": "x", "action": "y", "action_input": {}, "estimated_cost": 0.0},
    )
    assert r.status_code == 200
    assert r.json()["decision"] == "abstain"
    assert r.json()["reason"] == "max_steps_reached"


def test_cost_budget_forces_abstain(client: TestClient):
    r = client.post(
        "/v1/sessions",
        json={"task": "search flights", "max_steps": 10, "budget": 0.10},
    )
    sid = r.json()["session_id"]

    client.post(
        f"/v1/sessions/{sid}/step",
        json={
            "thought": "s",
            "action": "search_flights",
            "action_input": {},
            "estimated_cost": 0.05,
        },
    )
    client.post(
        f"/v1/sessions/{sid}/observation",
        json={"observation": "ok", "cost": 0.05, "is_error": False},
    )

    client.post(
        f"/v1/sessions/{sid}/step",
        json={
            "thought": "s2",
            "action": "search_flights",
            "action_input": {"retry": True},
            "estimated_cost": 0.05,
        },
    )
    client.post(
        f"/v1/sessions/{sid}/observation",
        json={"observation": "ok", "cost": 0.05, "is_error": False},
    )

    r = client.post(
        f"/v1/sessions/{sid}/step",
        json={
            "thought": "s3",
            "action": "search_flights",
            "action_input": {"retry": 2},
            "estimated_cost": 0.05,
        },
    )
    body = r.json()
    assert body["decision"] == "abstain"
    assert body["reason"] == "cost_budget_reached"
    assert body["threshold"] == 0.0
    assert "cost budget" in (body["abstention_message"] or "")


def test_end_session_writes_trajectory(client: TestClient, tmp_path):
    sid = _start(client)
    client.post(
        f"/v1/sessions/{sid}/step",
        json={
            "thought": "",
            "action": "search_flights",
            "action_input": {},
            "estimated_cost": 0.0,
        },
    )
    client.post(
        f"/v1/sessions/{sid}/observation",
        json={"observation": "ok", "cost": 0.0, "is_error": False},
    )
    r = client.post(
        f"/v1/sessions/{sid}/end",
        json={"outcome": "success", "final_answer": "done"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["outcome"] == "success"
    assert body["total_steps"] == 1
    assert body["trajectory_path"] is not None


def test_playbooks_endpoint_empty(client: TestClient):
    r = client.get("/v1/playbooks")
    assert r.status_code == 200
    assert r.json() == []


def test_suggest_playbooks_endpoint(client: TestClient):
    sid = _start(client)
    r = client.get(f"/v1/tasks/{sid}/playbooks")
    assert r.status_code == 200
    body = r.json()
    assert body["task"].startswith("book a flight")
    assert body["suggestions"] == []


def test_invalid_create_payload_returns_422(client: TestClient):
    r = client.post("/v1/sessions", json={"task": ""})
    assert r.status_code == 422


def test_validate_action_input_length(client: TestClient):
    sid = _start(client)
    r = client.post(
        f"/v1/sessions/{sid}/step",
        json={
            "thought": "",
            "action": "a" * 500,
            "action_input": {},
            "estimated_cost": 0.0,
        },
    )
    assert r.status_code == 422
