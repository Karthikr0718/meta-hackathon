from models import ToolSpec

# Your tool registry - based on your n8n workflows
TOOL_REGISTRY = {
    "job_search_api": ToolSpec(
        tool_id="job_search_api",
        name="Job Search API",
        description="Search for jobs by criteria (experience, skills, location)",
        cost_usd=2.0,
        latency_seconds=3.0,
        success_rate=0.95,
        prerequisites=[],
        fallback_tools=["job_search_api_v2"],
    ),
    "groq_infer": ToolSpec(
        tool_id="groq_infer",
        name="Groq LLM Inference",
        description="Run LLM inference to filter/process results",
        cost_usd=0.5,
        latency_seconds=2.0,
        success_rate=0.95,
        prerequisites=[],
        fallback_tools=[],
    ),
    "sheets_write": ToolSpec(
        tool_id="sheets_write",
        name="Google Sheets Write",
        description="Write results to Google Sheets",
        cost_usd=1.0,
        latency_seconds=2.0,
        success_rate=0.90,
        prerequisites=[],
        fallback_tools=["csv_write"],
    ),
    "db_fetch": ToolSpec(
        tool_id="db_fetch",
        name="Database Fetch",
        description="Fetch candidate data from database",
        cost_usd=1.0,
        latency_seconds=2.0,
        success_rate=0.95,
        prerequisites=[],
        fallback_tools=[],
    ),
    "gmail_fetch": ToolSpec(
        tool_id="gmail_fetch",
        name="Gmail Fetch",
        description="Fetch emails from Gmail",
        cost_usd=0.5,
        latency_seconds=1.0,
        success_rate=0.95,
        prerequisites=[],
        fallback_tools=[],
    ),
    "job_search_api_v2": ToolSpec(
        tool_id="job_search_api_v2",
        name="Job Search API V2 (Fallback)",
        description="Fallback job search with slower but more reliable service",
        cost_usd=2.5,
        latency_seconds=5.0,
        success_rate=0.98,
        prerequisites=[],
        fallback_tools=[],
    ),
    "csv_write": ToolSpec(
        tool_id="csv_write",
        name="CSV Write (Fallback)",
        description="Write results to CSV file instead of Google Sheets",
        cost_usd=0.2,
        latency_seconds=1.0,
        success_rate=1.0,
        prerequisites=[],
        fallback_tools=[],
    ),
}

def simulate_tool_call(tool_id: str, tool_input: dict, seed: int = 42):
    """
    Simulate a tool execution.
    
    Returns: (success, output, latency, cost)
    """
    import random
    random.seed(seed)
    
    if tool_id not in TOOL_REGISTRY:
        return False, None, 0.0, 0.0
    
    tool = TOOL_REGISTRY[tool_id]
    
    # Simulate success based on success_rate
    success = random.random() < tool.success_rate
    
    # Generate mock output based on tool
    output = None
    if success:
        if tool_id in ["job_search_api", "job_search_api_v2"]:
            output = {
                "jobs": [
                    {"id": f"job_{i}", "title": f"Software Engineer {i}", "company": f"Company {i}"}
                    for i in range(5)
                ]
            }
        elif tool_id == "groq_infer":
            output = {"filtered_jobs": 3, "result": "Filtered successfully"}
        elif tool_id == "sheets_write":
            output = {"status": "success", "rows_written": 5}
        elif tool_id == "csv_write":
            output = {"status": "success", "file": "results.csv", "rows_written": 5}
        elif tool_id == "db_fetch":
            output = {"candidate": {"name": "Karthik", "experience": "3 years", "skills": ["Python", "JS"]}}
        elif tool_id == "gmail_fetch":
            output = {"emails_count": 5, "unread": 2}
        else:
            output = {"status": "success"}
    
    return success, output, tool.latency_seconds, tool.cost_usd
