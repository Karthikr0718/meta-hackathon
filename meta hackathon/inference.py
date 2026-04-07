import os
import sys
from env import WorkflowOptimizationEnv
from models import Action
 
def get_tool_strategy(task_id):
   
    if task_id == "easy":
        return ["job_search_api"]
    elif task_id == "medium":
        return ["db_fetch", "job_search_api", "groq_infer"]
    elif task_id == "hard":
       
        return ["db_fetch", "job_search_api", "groq_infer"]
    return []
 
def run_task(env, task_id):
    """
    Run a single task with smart tool calling strategy.
    
    Returns: final_score
    """
    print(f"[START] {task_id}")
    
    # Reset environment for this task
    obs = env.reset(task_id)
    step_count = 0
    tools_to_call = get_tool_strategy(task_id)
    
    # Execute tool strategy
    for tool_id in tools_to_call:
        # Check if tool is available
        available_tool_ids = [t.tool_id for t in obs.available_tools]
        if tool_id not in available_tool_ids:
            continue
        
        # Check budget and time before calling
        if obs.budget_remaining <= 0.1 or obs.time_remaining <= 0.5:
            break
        
        # Call the tool
        obs, reward, done, info = env.step(
            Action(
                action_type="call_tool",
                tool_id=tool_id,
                tool_input={}
            )
        )
        
        step_count += 1
        
        # Log step with required format
        print(
            f"[STEP] step={step_count}, tool={tool_id}, "
            f"reward={reward.score:.3f}, budget_remaining={obs.budget_remaining:.2f}, "
            f"time_remaining={obs.time_remaining:.1f}"
        )
        
        # If budget or time is critically low, stop calling tools
        if obs.budget_remaining <= 0.1 or obs.time_remaining <= 0.5:
            break
    
    # Submit final result
    obs, reward, done, info = env.step(
        Action(action_type="submit_result")
    )
    
    step_count += 1
    final_score = info.get("task_score", 0.0)
    
    # Log submission step
    print(
        f"[STEP] step={step_count}, tool=submit_result, "
        f"reward={reward.score:.3f}, budget_remaining={obs.budget_remaining:.2f}, "
        f"time_remaining={obs.time_remaining:.1f}"
    )
    
    # Log task completion with required format
    print(f"[END] {task_id}, final_score={final_score:.3f}")
    print()  # Blank line between tasks
    
    return final_score
 
def main():
    """Main execution: run all 3 tasks and collect scores."""
    
    # Load environment variables
    api_base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    model_name = os.getenv("MODEL_NAME", "meta-llama/Llama-2-7b-chat-hf")
    hf_token = os.getenv("HF_TOKEN", "")
    
    # Initialize environment
    env = WorkflowOptimizationEnv()
    
    # Run all 3 tasks
    scores = {}
    for task_id in ["easy", "medium", "hard"]:
        score = run_task(env, task_id)
        scores[task_id] = score
    
    # Print summary (optional, for debugging)
    # print("="*70)
    # print("FINAL SUMMARY:")
    # print("-"*70)
    # for task_id, score in scores.items():
    #     print(f"  {task_id.upper():10s}: {score:.3f}")
    # print("="*70)
    
    # Exit with success code
    sys.exit(0)
 
if __name__ == "__main__":
    main()
