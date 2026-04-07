from typing import Tuple, Optional, Dict, Any
from models import Observation, Action, Reward, TaskInfo, ExecutionRecord
from tools import simulate_tool_call, TOOL_REGISTRY
from tasks import TASK_CONFIGS
from graders import grade_easy_task, grade_medium_task, grade_hard_task

class WorkflowOptimizationEnv:
    """
    OpenEnv-compatible environment for AI agent workflow optimization.
    
    Agents must orchestrate tool calls under budget and time constraints
    to complete various tasks efficiently.
    """
    
    def __init__(self):
        self.current_task: Optional[TaskInfo] = None
        self.current_step = 0
        self.execution_history = []
        self.current_state = {}
        self.time_spent = 0.0
        self.cost_spent = 0.0
        self.seed_value = 42
    
    def reset(self, task_id: str = "easy") -> Observation:
        """
        Reset environment for a new task.
        
        Args:
            task_id: "easy", "medium", or "hard"
            
        Returns:
            Initial Observation
        """
        if task_id not in TASK_CONFIGS:
            raise ValueError(f"Unknown task_id: {task_id}")
        
        self.current_task = TASK_CONFIGS[task_id]
        self.current_step = 0
        self.execution_history = []
        self.current_state = {}
        self.time_spent = 0.0
        self.cost_spent = 0.0
        
        return self._get_observation()
    
    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        """
        Execute one step of the workflow.
        
        Args:
            action: Agent's decision
            
        Returns:
            (observation, reward, done, info)
        """
        if self.current_task is None:
            raise RuntimeError("Must call reset() before step()")
        
        done = False
        reward = Reward(score=0.0)
        info = {}
        
        # Handle different action types
        if action.action_type == "call_tool":
            reward = self._execute_tool_call(action)
        elif action.action_type == "submit_result":
            done = True
            reward = self._grade_task()
            info["task_score"] = reward.score
        elif action.action_type == "abort":
            done = True
            reward = Reward(score=0.0, breakdown="Aborted by agent")
            info["task_score"] = 0.0
        else:
            reward = Reward(score=-0.02, breakdown=f"Unknown action type: {action.action_type}")
        
        self.current_step += 1
        
        # Check episode termination conditions
        if self.current_step >= self.current_task.max_steps:
            done = True
        if self.cost_spent >= self.current_task.budget_limit:
            done = True
        if self.time_spent >= self.current_task.time_limit:
            done = True
        
        obs = self._get_observation()
        return obs, reward, done, info
    
    def state(self) -> Dict[str, Any]:
        """Return current workflow state snapshot."""
        return {
            "task_id": self.current_task.task_id if self.current_task else None,
            "current_step": self.current_step,
            "time_spent": self.time_spent,
            "cost_spent": self.cost_spent,
            "execution_history": [r.dict() for r in self.execution_history],
            "current_state": self.current_state,
        }
    
    def _execute_tool_call(self, action: Action) -> Reward:
        """Simulate tool execution and update state."""
        tool_id = action.tool_id
        tool_input = action.tool_input or {}
        
        if tool_id not in TOOL_REGISTRY:
            return Reward(score=-0.05, breakdown=f"Tool not found: {tool_id}")
        
        tool_spec = TOOL_REGISTRY[tool_id]
        
        # Check prerequisites
        completed_tools = {r.tool_id for r in self.execution_history if r.status == "success"}
        missing_prereqs = [p for p in tool_spec.prerequisites if p not in completed_tools]
        if missing_prereqs:
            return Reward(score=-0.05, breakdown=f"Unmet prerequisites: {missing_prereqs}")
        
        # Simulate execution
        success, output, latency, cost = simulate_tool_call(tool_id, tool_input, self.seed_value)
        
        # Record execution
        record = ExecutionRecord(
            tool_id=tool_id,
            status="success" if success else "failure",
            input=tool_input,
            output=output,
            cost=cost,
            latency=latency,
            timestamp=self.time_spent
        )
        self.execution_history.append(record)
        
        # Update tracking
        self.time_spent += latency
        self.cost_spent += cost
        
        if success:
            self.current_state.update(output or {})
            return Reward(score=0.05, breakdown=f"Tool {tool_id} succeeded")
        else:
            return Reward(score=-0.02, breakdown=f"Tool {tool_id} failed")
    
    def _grade_task(self) -> Reward:
        """Grade task completion and return final reward."""
        grader_fn = {
            "easy": grade_easy_task,
            "medium": grade_medium_task,
            "hard": grade_hard_task,
        }[self.current_task.difficulty]
        
        score = grader_fn(
            self.execution_history,
            self.current_state,
            self.current_task.budget_limit,
            self.current_task.time_limit
        )
        
        return Reward(
            score=score,
            breakdown=f"Final grade for {self.current_task.difficulty} task"
        )
    
    def _get_observation(self) -> Observation:
        """Build current observation for agent."""
        if self.current_task is None:
            raise RuntimeError("No active task")
        
        return Observation(
            task_id=self.current_task.task_id,
            task_description=self.current_task.description,
            available_tools=self.current_task.tools_config,
            budget_remaining=max(0, self.current_task.budget_limit - self.cost_spent),
            time_remaining=max(0, self.current_task.time_limit - self.time_spent),
            execution_history=self.execution_history.copy(),
            current_state=self.current_state.copy(),
            max_steps=self.current_task.max_steps,
            current_step=self.current_step,
        )
