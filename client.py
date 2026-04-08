from dataclasses import dataclass
from typing import Any, Dict
from openenv.core import EnvClient
from incident_env.models import IncidentEnvAction, IncidentEnvObservation, IncidentEnvState


@dataclass
class StepResult:
    observation: IncidentEnvObservation
    reward: float
    done: bool
    info: dict


class IncidentEnvClient(EnvClient):
    Action = IncidentEnvAction
    Observation = IncidentEnvObservation

    def _step_payload(self, action: IncidentEnvAction) -> Dict[str, Any]:
        return {"action": action.model_dump(exclude_none=True)}

    def _parse_result(self, data: Dict[str, Any]) -> StepResult:
        obs_data = data.get("observation", data)
        reward = float(data.get("reward", 0.0))
        done = bool(data.get("done", False))
        obs = IncidentEnvObservation(**obs_data)
        return StepResult(observation=obs, reward=reward, done=done, info={})

    def _parse_state(self, data: Dict[str, Any]) -> IncidentEnvState:
        safe = {
            "episode_id": data.get("episode_id", "unknown"),
            "task_id": data.get("task_id", ""),
            "step_count": data.get("step_count", 0),
            "queries_made": data.get("queries_made", []),
            "severity_set": data.get("severity_set", False),
            "runbooks_applied": data.get("runbooks_applied", []),
            "resolved": data.get("resolved", False),
            "ground_truth_root_cause": data.get("ground_truth_root_cause", ""),
            "ground_truth_runbook": data.get("ground_truth_runbook", ""),
            "score_components": data.get("score_components", {}),
        }
        return IncidentEnvState(**safe)
