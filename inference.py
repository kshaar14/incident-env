import asyncio
import json
import os
import textwrap
from typing import List, Optional
from openai import OpenAI
from incident_env.client import IncidentEnvClient
from incident_env.models import IncidentEnvAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN")
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "https://shaark14-incident-env.hf.space")

TASKS = [
    "task_easy_payment_timeout",
    "task_medium_checkout_cascade",
    "task_hard_auth_degradation",
]
MAX_STEPS = 12
BENCHMARK = "incident_env"

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert SRE (Site Reliability Engineer) responding to production incidents.
    You investigate by querying tools, then apply the correct runbook, then resolve.

    Available action_types:
    - query_tool: query logs/metrics/traces/topology
    - set_severity: set incident severity (P1/P2/P3)
    - apply_runbook: apply a runbook by ID
    - escalate: escalate to senior SRE
    - resolve: close the incident with your root cause diagnosis

    Always respond with ONLY valid JSON matching one of these schemas:
    {"action_type":"query_tool","tool_name":"logs","tool_args":{"service":"payment-api","window":"5m"}}
    {"action_type":"set_severity","severity":"P1"}
    {"action_type":"apply_runbook","runbook_id":"runbook_db_pool_scale"}
    {"action_type":"resolve","root_cause":"your diagnosis here","resolution_note":"steps taken"}

    Strategy: query 2-3 tools first to gather evidence, then set severity, apply the correct runbook, then resolve.
    No explanation. JSON only.
""").strip()


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    err = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)

def log_end(success, steps, score, rewards):
    r = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={r}", flush=True)


def get_action(client, obs, step, history) -> tuple[IncidentEnvAction, str]:
    prompt = textwrap.dedent(f"""
        ALERT: {obs.alert_summary}
        Last tool output: {obs.tool_output or 'None'}
        Feedback: {obs.step_feedback}
        Status: {obs.incident_status} | Step: {obs.elapsed_steps}
        Available tools: {obs.available_tools}
        Available runbooks: {obs.available_runbooks}
        History: {history[-4:]}

        Respond with JSON action only.
    """).strip()

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.2,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        return IncidentEnvAction(**data), raw
    except Exception as e:
        fallback = IncidentEnvAction(
            action_type="resolve",
            root_cause="parse error fallback",
            resolution_note=str(e),
        )
        return fallback, f"error:{e}"


async def run_task(task_id: str, client: OpenAI) -> float:
    log_start(task_id, BENCHMARK, MODEL_NAME)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        async with IncidentEnvClient(base_url=HF_SPACE_URL) as env:
            result = await env.reset(task_id=task_id)
            obs = result.observation
            history = []

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                action, raw = get_action(client, obs, step, history)
                result = await env.step(action)
                obs = result.observation
                reward = result.reward or 0.0
                done = result.done

                rewards.append(reward)
                steps_taken = step
                history.append(f"step={step} type={action.action_type} reward={reward:.2f}")
                log_step(step, raw[:120].replace("\n", " "), reward, done, None)

                if done:
                    break

        score = min(max(sum(rewards), 0.0), 1.0)
        success = score >= 0.3

    except Exception as e:
        print(f"[DEBUG] Task {task_id} error: {e}", flush=True)
        log_end(False, steps_taken, 0.0, rewards)
        return 0.0

    log_end(success, steps_taken, score, rewards)
    return score


async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    total = 0.0
    for task_id in TASKS:
        score = await run_task(task_id, client)
        total += score
        print(f"[DEBUG] {task_id} score={score:.3f}", flush=True)
    print(f"[DEBUG] mean_score={total/len(TASKS):.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
