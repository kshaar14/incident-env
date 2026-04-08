ROOT_CAUSE_KEYWORDS = {
    "database_connection_pool_exhausted": ["pool", "connection", "db", "database", "exhausted", "queue"],
    "inventory_service_memory_leak": ["memory", "heap", "oom", "inventory", "gc", "leak"],
    "certificate_expiry_partial_rollout": ["cert", "certificate", "ssl", "tls", "expir", "partial", "rollout"],
}


def _fuzzy_match_root_cause(stated: str, ground_truth: str) -> float:
    keywords = ROOT_CAUSE_KEYWORDS.get(ground_truth, [])
    if not keywords or not stated:
        return 0.0
    stated_lower = stated.lower()
    hits = sum(1 for k in keywords if k in stated_lower)
    return hits / len(keywords)


def grade_episode(state, task: dict) -> float:
    score = 0.0

    # 0.0–0.3: queried relevant tools
    relevant = task.get("relevant_tool_keys", [])
    if relevant:
        tools_hit = sum(1 for q in state.queries_made if q in relevant)
        score += min(tools_hit / len(relevant), 1.0) * 0.3

    # 0.0–0.1: set severity
    if state.severity_set:
        score += 0.1

    # 0.0–0.3: applied correct runbook
    if task["ground_truth_runbook"] in state.runbooks_applied:
        score += 0.3

    # 0.0–0.3: correct root cause stated
    if state.resolved:
        rc_match = _fuzzy_match_root_cause(
            state.score_components.get("stated_root_cause", ""),
            task["ground_truth_root_cause"],
        )
        score += rc_match * 0.3

    return round(min(score, 1.0), 3)
