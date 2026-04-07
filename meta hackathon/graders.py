from typing import List, Dict, Any
from models import ExecutionRecord

def grade_easy_task(
    execution_history: List[ExecutionRecord],
    current_state: Dict[str, Any],
    budget: float = 5.0,
    time_limit: float = 30.0
) -> float:
    """Grade easy task: simple job search."""
    task_complete = len(current_state.get("jobs", [])) >= 3
    cost_spent = sum(r.cost for r in execution_history)
    time_spent = sum(r.latency for r in execution_history)
    
    completion_score = 1.0 if task_complete else 0.0
    cost_penalty = max(0, (cost_spent - budget) / budget) * 0.3
    time_penalty = max(0, (time_spent - time_limit) / time_limit) * 0.2
    
    score = completion_score - cost_penalty - time_penalty
    return max(0.0, min(1.0, score))

def grade_medium_task(
    execution_history: List[ExecutionRecord],
    current_state: Dict[str, Any],
    budget: float = 10.0,
    time_limit: float = 60.0
) -> float:
    """Grade medium task: multi-step enrichment workflow."""
    steps_completed = len([r for r in execution_history if r.status == "success"])
    all_steps_ok = steps_completed >= 4
    
    cost_spent = sum(r.cost for r in execution_history)
    time_spent = sum(r.latency for r in execution_history)
    
    completion_score = 1.0 if all_steps_ok else (steps_completed / 4.0)
    
    cost_ratio = cost_spent / budget
    time_ratio = time_spent / time_limit
    
    efficiency_score = 1.0 / (1.0 + cost_ratio + time_ratio)
    
    score = completion_score * 0.6 + efficiency_score * 0.4
    return max(0.0, min(1.0, score))

def grade_hard_task(
    execution_history: List[ExecutionRecord],
    current_state: Dict[str, Any],
    budget: float = 4.0,
    time_limit: float = 45.0
) -> float:
    """
    Grade hard task: complex workflow with cascading failures.
    
    Score based on:
    1. How many core steps completed (fetch, search, infer)
    2. Cost efficiency
    3. Time efficiency
    4. Resilience (using fallbacks)
    """
    cost_spent = sum(r.cost for r in execution_history)
    time_spent = sum(r.latency for r in execution_history)
    
    # Count successful tool calls (exclude submit_result)
    successful_tools = [r for r in execution_history if r.status == "success" and "submit" not in r.tool_id.lower()]
    
    # Measure completion of core steps
    fetch_done = any("db" in r.tool_id.lower() or "fetch" in r.tool_id.lower() 
                     for r in successful_tools)
    search_done = any("search" in r.tool_id.lower() or "job" in r.tool_id.lower() 
                      for r in successful_tools)
    infer_done = any("infer" in r.tool_id.lower() or "groq" in r.tool_id.lower()
                     for r in successful_tools)
    
    completed_core = sum([fetch_done, search_done, infer_done])
    
    # Base completion score (0.33 per core step)
    completion_score = completed_core / 3.0
    
    # If no steps completed, return 0
    if completed_core == 0:
        return 0.0
    
    # Penalties for budget/time overrun
    budget_overrun = max(0, (cost_spent - budget) / budget)
    time_overrun = max(0, (time_spent - time_limit) / time_limit)
    
    # Bonus for using fallbacks (resilience)
    fallback_count = sum(1 for r in successful_tools if "v2" in r.tool_id.lower())
    resilience_bonus = min(0.1, fallback_count * 0.05)
    
    # Final score calculation
    # Base: 0.5 weight on completion
    # Penalties: 0.2 for budget, 0.15 for time
    # Bonus: Up to 0.1 for resilience
    score = (completion_score * 0.5 
             - budget_overrun * 0.2 
             - time_overrun * 0.15 
             + resilience_bonus)
    
    # Ensure score is in valid range
    return max(0.0, min(1.0, score))