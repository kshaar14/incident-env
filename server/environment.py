import uuid
from openenv.core.env_server import Environment
from incident_env.models import IncidentEnvAction, IncidentEnvObservation, IncidentEnvState
from incident_env.server.incidents import INCIDENTS
from incident_env.server.graders import grade_episode

MAX_STEPS = 15
STEP_PENALTY = 0.01


def _find_tool_response(tool_responses: dict, tool_name: str, service: str, window: str):
    """Match tool response by tool+service, ignoring window mismatch."""
    # exact match first
    exact = f"{tool_name}:{service}:{window}"
    if exact in tool_responses:
        return exact, tool_responses[exact]
    # fuzzy: same tool+service, any window
    prefix = f"{tool_name}:{service}:"
    for key, val in tool_responses.items():
        if key.startswith(prefix):
            return key, val
    return None, None


class IncidentEnvEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self._state: IncidentEnvState = IncidentEnvState(episode_id="uninitialised", task_id="")
        self._task: dict = INCIDENTS["task_easy_payment_timeout"]

    def reset(self, task_id: str = "task_easy_payment_timeout") -> IncidentEnvObservation:
        task_id = task_id if task_id in INCIDENTS else "task_easy_payment_timeout"
        self._task = INCIDENTS[task_id]
        self._state = IncidentEnvState(
            episode_id=str(uuid.uuid4()),
            task_id=task_id,
            ground_truth_root_cause=self._task["ground_truth_root_cause"],
            ground_truth_runbook=self._task["ground_truth_runbook"],
        )
        return IncidentEnvObservation(
            reward=0.0,
            done=False,
            alert_summary=self._task["alert"],
            tool_output=None,
            available_tools=["logs", "metrics", "traces", "topology"],
            available_runbooks=list(self._task["runbooks"].keys()),
            current_severity=None,
            step_feedback="Incident opened. Investigate using tools, then resolve.",
            incident_status="open",
            elapsed_steps=0,
        )

    def step(self, action: IncidentEnvAction) -> IncidentEnvObservation:
        self._state.step_count += 1
        reward = -STEP_PENALTY
        done = False
        tool_output = None
        feedback = ""

        if action.action_type == "query_tool":
            svc = (action.tool_args or {}).get("service", "")
            win = (action.tool_args or {}).get("window", "")
            matched_key, result = _find_tool_response(
                self._task["tool_responses"], action.tool_name, svc, win
            )
            if result:
                tool_output = result
                if matched_key not in self._state.queries_made:
                    self._state.queries_made.append(matched_key)
                reward += 0.05
                feedback = f"Tool '{action.tool_name}' returned data for service='{svc}'."
            else:
                available = list(self._task["tool_responses"].keys())
                tool_output = (
                    f"No data for tool='{action.tool_name}' service='{svc}'. "
                    f"Available services: {list(set(k.split(':')[1] for k in available))}"
                )
                feedback = "No data found. Check the service name."

        elif action.action_type == "set_severity":
            self._state.severity_set = True
            reward += 0.05
            feedback = f"Severity set to {action.severity}."

        elif action.action_type == "apply_runbook":
            rb = action.runbook_id or ""
            if rb in self._task["runbooks"]:
                self._state.runbooks_applied.append(rb)
                correct = rb == self._task["ground_truth_runbook"]
                reward += 0.3 if correct else -0.1
                feedback = (
                    f"Runbook '{rb}' applied: {self._task['runbooks'][rb]}. "
                    + ("Correct remediation." if correct else "May not address root cause.")
                )
            else:
                feedback = f"Unknown runbook '{rb}'. Check available_runbooks."

        elif action.action_type == "resolve":
            self._state.resolved = True
            self._state.score_components["stated_root_cause"] = action.root_cause or ""
            final_score = grade_episode(self._state, self._task)
            reward += final_score * 0.5
            done = True
            feedback = f"Incident resolved. Grader score: {final_score:.3f}."

        elif action.action_type == "escalate":
            feedback = "Escalated to senior SRE. Continue investigating."

        if self._state.step_count >= MAX_STEPS and not done:
            done = True
            feedback += " Max steps reached."

        return IncidentEnvObservation(
            reward=round(reward, 4),
            done=done,
            alert_summary=self._task["alert"],
            tool_output=tool_output,
            available_tools=["logs", "metrics", "traces", "topology"],
            available_runbooks=list(self._task["runbooks"].keys()),
            current_severity=action.severity if action.action_type == "set_severity" else None,
            step_feedback=feedback,
            incident_status="resolved" if self._state.resolved else "open",
            elapsed_steps=self._state.step_count,
        )

    @property
    def state(self) -> IncidentEnvState:
        return self._state
