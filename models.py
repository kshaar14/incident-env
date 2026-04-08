from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict, Any
from openenv.core import EnvClient


class IncidentEnvAction(BaseModel):
    action_type: Literal["query_tool", "set_severity", "apply_runbook", "escalate", "resolve"]
    tool_name: Optional[str] = Field(None, description="One of: logs, metrics, traces, topology")
    tool_args: Optional[Dict[str, str]] = Field(None, description='e.g. {"service":"payment-api","window":"5m"}')
    severity: Optional[str] = Field(None, description="P1, P2, or P3")
    runbook_id: Optional[str] = Field(None, description="ID from available_runbooks list")
    root_cause: Optional[str] = Field(None, description="Root cause diagnosis (used with resolve)")
    resolution_note: Optional[str] = Field(None, description="Summary of remediation steps taken")


class IncidentEnvObservation(BaseModel):
    # OpenEnv framework requires reward on the Observation model
    reward: float = 0.0
    done: bool = False

    alert_summary: str
    tool_output: Optional[str] = None
    available_tools: List[str]
    available_runbooks: List[str]
    current_severity: Optional[str] = None
    step_feedback: str
    incident_status: str
    elapsed_steps: int


class IncidentEnvState(BaseModel):
    episode_id: str = "uninitialised"
    task_id: str = ""
    step_count: int = 0
    queries_made: List[str] = Field(default_factory=list)
    severity_set: bool = False
    runbooks_applied: List[str] = Field(default_factory=list)
    resolved: bool = False
    ground_truth_root_cause: str = ""
    ground_truth_runbook: str = ""
    score_components: Dict[str, Any] = Field(default_factory=dict)


class IncidentEnvClient(EnvClient):
    Action = IncidentEnvAction
    Observation = IncidentEnvObservation
