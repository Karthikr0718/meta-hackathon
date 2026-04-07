from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal

class ToolSpec(BaseModel):
    """Specification of a tool the agent can call."""
    tool_id: str
    name: str
    description: str
    cost_usd: float
    latency_seconds: float
    success_rate: float = Field(..., ge=0.0, le=1.0)
    prerequisites: List[str] = Field(default_factory=list)
    fallback_tools: List[str] = Field(default_factory=list)

class ExecutionRecord(BaseModel):
    """Record of a single tool execution."""
    tool_id: str
    status: Literal["success", "failure"]
    input: Dict[str, Any]
    output: Optional[Dict[str, Any]] = None
    cost: float
    latency: float
    timestamp: float

class Observation(BaseModel):
    """What the agent sees at each step."""
    task_id: str
    task_description: str
    available_tools: List[ToolSpec]
    budget_remaining: float
    time_remaining: float
    execution_history: List[ExecutionRecord] = Field(default_factory=list)
    current_state: Dict[str, Any] = Field(default_factory=dict)
    max_steps: int = 20
    current_step: int = 0

class Action(BaseModel):
    """What the agent decides to do."""
    action_type: Literal["call_tool", "retry_tool", "skip_tool", "escalate", "submit_result", "abort"]
    tool_id: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    fallback_tool_id: Optional[str] = None
    escalation_reason: Optional[str] = None

class Reward(BaseModel):
    """Reward signal."""
    score: float = Field(..., ge=0.0, le=1.0)
    breakdown: str = ""

class TaskInfo(BaseModel):
    """Task metadata."""
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    budget_limit: float
    time_limit: float
    tools_config: List[ToolSpec] = Field(default_factory=list)
    max_steps: int = 20
