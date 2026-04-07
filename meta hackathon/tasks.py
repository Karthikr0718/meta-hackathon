from models import TaskInfo, ToolSpec
from tools import TOOL_REGISTRY

TASK_CONFIGS = {
    "easy": TaskInfo(
        task_id="easy",
        difficulty="easy",
        description="Find a job for Karthik matching: 2+ years experience, tech stack (Python, JS), location (Bangalore or remote). Budget: $5. Time: 30 seconds.",
        budget_limit=5.0,
        time_limit=30.0,
        tools_config=[
            TOOL_REGISTRY["job_search_api"],
            TOOL_REGISTRY["groq_infer"],
        ],
        max_steps=20,
    ),
    "medium": TaskInfo(
        task_id="medium",
        difficulty="medium",
        description="Multi-step workflow: fetch candidate data, search jobs, generate pitch, write to sheets. Budget: $10. Time: 60 seconds.",
        budget_limit=10.0,
        time_limit=60.0,
        tools_config=[
            TOOL_REGISTRY["db_fetch"],
            TOOL_REGISTRY["job_search_api"],
            TOOL_REGISTRY["groq_infer"],
            TOOL_REGISTRY["sheets_write"],
        ],
        max_steps=20,
    ),
    "hard": TaskInfo(
        task_id="hard",
        difficulty="hard",
        description="Complex workflow with cascading failures, adversarial conditions, tight budget and time. Budget: $4. Time: 45 seconds.",
        budget_limit=4.0,
        time_limit=45.0,
        tools_config=[
            TOOL_REGISTRY["db_fetch"],
            TOOL_REGISTRY["job_search_api"],
            TOOL_REGISTRY["job_search_api_v2"],
            TOOL_REGISTRY["groq_infer"],
            TOOL_REGISTRY["sheets_write"],
            TOOL_REGISTRY["csv_write"],
            TOOL_REGISTRY["gmail_fetch"],
        ],
        max_steps=20,
    ),
}
