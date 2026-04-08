import asyncio
import json
import os
import textwrap
from typing import List
from openai import OpenAI
import httpx
from incident_env.models import IncidentEnvAction, IncidentEnvObservation

# CRITICAL: use API_BASE_URL and API_KEY exactly as injected by the validator
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN", "dummy")
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "https://shaark14-incident-env.hf.space")

TASKS = [
    "task_easy_payment_timeout",
    "task_medium_checkout_cascade",
    "task_hard_auth_degradation",
]
MAX_STEPS = 12
BENCHMARK = "incident_env"

SYSTEM_PROMPT = """You are an expert SRE responding to production incidents.
Investigate by querying tools, apply the correct runbook, then resolve.

Respond ONLY with valid JSON. Examples:
{"action_type":"query_tool","tool_name":"logs","tool_args":{"service":"payment-api","window":"5m"}}
{"action_type":"query_tool","tool_name":"metrics","tool_args":{"service":"payment-api","window":"5m"}}
{"action_type":"set_severity","severity":"P1"}
{"action_type":"apply_runbook","runbook_id":"runbook_db_pool_scale"}
{"action_type":"resolve","root_cause":"database connection pool exhausted","resolution_note":"scaled pool"}

No explanation. JSON only."""


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    r = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={r}", flush=True)


def get_action(client: OpenAI, obs: IncidentEnvObservation, history: List[str]):
    prompt = f"""Alert: {obs.alert_summary}
Last tool output: {obs.tool_output or 'None'}
Feedback: {obs.step_feedback}
Status: {obs.incident_status} | Steps: {obs.elapsed_steps}
Available tools: {obs.available_tools}
Available runbooks: {obs.available_runbooks}
History: {history[-3:]}

Respond with JSON action only."""

    try:
        # This call MUST go through API_BASE_URL so the proxy sees it
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
            temperature=0.2,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        return IncidentEnvAction(**data), raw
    except Exception as e:
        action = IncidentEnvAction(
            action_type="escalate",
        )
        return action, f"fallback:{e}"


async def run_task(task_id: str, client: OpenAI) -> float:
    log_start(task_id, BENCHMARK, MODEL_NAME)
    rewards: List[float] = []
    steps_taken = 0
    base = HF_SPACE_URL.rstrip("/")

    try:
        async with httpx.AsyncClient(timeout=60) as http:
            # Reset
            r = await http.post(
                f"{base}/reset",
                json={"task_id": task_id},
                headers={"Content-Type": "application/json"},
            )
            data = r.json()
            obs = IncidentEnvObservation(**data["observation"])
            done = data.get("done", False)
            history = []

            for step in range(1, MAX_STEPS + 1):
                if done:
                    break

                action, raw = get_action(client, obs, history)

                r = await http.post(
                    f"{base}/step",
                    json={"action": action.model_dump(exclude_none=True)},
                    headers={"Content-Type": "application/json"},
                )
                data = r.json()
                obs = IncidentEnvObservation(**data["observation"])
                reward = float(data.get("reward", 0.0))
                done = bool(data.get("done", False))

                rewards.append(reward)
                steps_taken = step
                history.append(f"step={step} {action.action_type} reward={reward:.2f}")
                log_step(step, raw[:100].replace("\n", " "), reward, done, None)

                if done:
                    break

        score = min(max(sum(rewards), 0.0), 1.0)
        success = score >= 0.3
        log_end(success, steps_taken, score, rewards)
        return score

    except Exception as e:
        print(f"[DEBUG] Task {task_id} error: {e}", flush=True)
        log_end(False, steps_taken, 0.0, rewards)
        return 0.0


async def main():
    # Initialize OpenAI client with API_BASE_URL and API_KEY from environment
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )
    total = 0.0
    for task_id in TASKS:
        score = await run_task(task_id, client)
        total += score
        print(f"[DEBUG] {task_id} score={score:.3f}", flush=True)
    print(f"[DEBUG] mean_score={total/len(TASKS):.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
