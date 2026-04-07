#!/usr/bin/env python3
"""
OpenAI-Enhanced Inference Engine for OpenEnv
Uses OpenAI's API for intelligent tool selection and workflow optimization.
"""

import os
import json
import subprocess
import sys
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

if not OPENAI_API_KEY:
    print("ERROR: OPENAI_API_KEY environment variable not set")
    sys.exit(1)

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)

class ToolCall(BaseModel):
    """Represents a tool call request from the LLM."""
    tool_name: str = Field(..., description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(..., description="Arguments for the tool")

class AgentResponse(BaseModel):
    """Represents the LLM's response."""
    thought: str = Field(..., description="The agent's reasoning")
    tool_calls: List[ToolCall] = Field(default=[], description="Tools to execute")

class ToolRegistry:
    """Registry of available tools."""
    
    def __init__(self):
        self.tools = {}
    
    def register(self, name: str, description: str, schema: Dict[str, Any]):
        """Register a tool with the registry."""
        self.tools[name] = {
            "name": name,
            "description": description,
            "parameters": schema
        }
    
    def get_tools_spec(self) -> List[Dict[str, Any]]:
        """Get tools specification for OpenAI API."""
        return list(self.tools.values())

class OpenEnvAgent:
    """OpenAI-powered agent for workflow optimization."""
    
    def __init__(self):
        self.tool_registry = ToolRegistry()
        self.setup_tools()
    
    def setup_tools(self):
        """Register available tools."""
        # Tool: execute_command
        self.tool_registry.register(
            name="execute_command",
            description="Execute a CLI command on the system",
            schema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The CLI command to execute"
                    },
                    "requires_approval": {
                        "type": "boolean",
                        "description": "Whether this command requires user approval"
                    }
                },
                "required": ["command", "requires_approval"]
            }
        )
        
        # Tool: read_file
        self.tool_registry.register(
            name="read_file",
            description="Read the contents of a file",
            schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Starting line number (1-based)"
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Ending line number (1-based)"
                    }
                },
                "required": ["path"]
            }
        )
        
        # Tool: write_to_file
        self.tool_registry.register(
            name="write_to_file",
            description="Write content to a file",
            schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    }
                },
                "required": ["path", "content"]
            }
        )
        
        # Tool: replace_in_file
        self.tool_registry.register(
            name="replace_in_file",
            description="Replace content in a file using SEARCH/REPLACE blocks",
            schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to modify"
                    },
                    "diff": {
                        "type": "string",
                        "description": "SEARCH/REPLACE blocks defining the changes"
                    }
                },
                "required": ["path", "diff"]
            }
        )
        
        # Tool: search_files
        self.tool_registry.register(
            name="search_files",
            description="Perform regex search across files",
            schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to search in"
                    },
                    "regex": {
                        "type": "string",
                        "description": "Regular expression pattern to search for"
                    },
                    "file_pattern": {
                        "type": "string",
                        "description": "File pattern to filter files (e.g., '*.ts')"
                    }
                },
                "required": ["path", "regex"]
            }
        )
        
        # Tool: list_files
        self.tool_registry.register(
            name="list_files",
            description="List files in a directory",
            schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the directory"
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to list files recursively"
                    }
                },
                "required": ["path"]
            }
        )
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the agent."""
        return """
You are an AI agent tasked with optimizing workflows and solving tasks using available tools.

Available Tools:
- execute_command: Execute CLI commands on the system
- read_file: Read file contents
- write_to_file: Write content to files
- replace_in_file: Replace content in files using SEARCH/REPLACE blocks
- search_files: Perform regex searches across files
- list_files: List files in directories

Guidelines:
1. Use tools one at a time
2. Wait for tool results before proceeding
3. Use SEARCH/REPLACE blocks for file modifications
4. Provide clear, actionable responses
5. Always consider the task context
"""
    
    def call_llm(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]]) -> AgentResponse:
        """Call the LLM with the given messages and tools."""
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.1
            )
            
            # Parse the response
            content = response.choices[0].message
            
            # Extract tool calls if any
            tool_calls = []
            if hasattr(content, 'tool_calls') and content.tool_calls:
                for tool_call in content.tool_calls:
                    tool_calls.append(ToolCall(
                        tool_name=tool_call.function.name,
                        arguments=json.loads(tool_call.function.arguments)
                    ))
            
            return AgentResponse(
                thought=content.content or "Proceeding with task...",
                tool_calls=tool_calls
            )
            
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return AgentResponse(
                thought=f"Error: {str(e)}",
                tool_calls=[]
            )
    
    def execute_tool_call(self, tool_call: ToolCall) -> str:
        """Execute a tool call and return the result."""
        tool_name = tool_call.tool_name
        args = tool_call.arguments
        
        print(f"\n🤖 Agent: Executing {tool_name}...")
        print(f"Arguments: {args}")
        
        if tool_name == "execute_command":
            try:
                result = subprocess.run(
                    args["command"],
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                output = f"Exit Code: {result.returncode}\n"
                if result.stdout:
                    output += f"STDOUT:\n{result.stdout}\n"
                if result.stderr:
                    output += f"STDERR:\n{result.stderr}\n"
                
                return output
                
            except subprocess.TimeoutExpired:
                return "Error: Command timed out after 5 minutes"
            except Exception as e:
                return f"Error executing command: {str(e)}"
        
        elif tool_name == "read_file":
            try:
                with open(args["path"], 'r') as f:
                    lines = f.readlines()
                
                start = args.get("start_line", 1)
                end = args.get("end_line", len(lines))
                
                content = "".join(lines[start-1:end])
                return f"File content ({start}-{end}):\n{content}"
                
            except Exception as e:
                return f"Error reading file: {str(e)}"
        
        elif tool_name == "write_to_file":
            try:
                with open(args["path"], 'w') as f:
                    f.write(args["content"])
                return f"Successfully wrote to {args['path']}"
            except Exception as e:
                return f"Error writing file: {str(e)}"
        
        elif tool_name == "replace_in_file":
            try:
                # This is a simplified implementation
                # In practice, you'd need to parse SEARCH/REPLACE blocks
                return "File modification would be handled by the system"
            except Exception as e:
                return f"Error modifying file: {str(e)}"
        
        elif tool_name == "search_files":
            try:
                # This is a simplified implementation
                # In practice, you'd need to implement the search logic
                return "Search results would be returned here"
            except Exception as e:
                return f"Error searching files: {str(e)}"
        
        elif tool_name == "list_files":
            try:
                path = args["path"]
                recursive = args.get("recursive", False)
                
                if recursive:
                    files = []
                    for root, dirs, filenames in os.walk(path):
                        for f in filenames:
                            files.append(os.path.join(root, f))
                    return f"Files in {path} (recursive):\n" + "\n".join(files)
                else:
                    files = os.listdir(path)
                    return f"Files in {path}:\n" + "\n".join(files)
                    
            except Exception as e:
                return f"Error listing files: {str(e)}"
        
        else:
            return f"Unknown tool: {tool_name}"
    
    def run_workflow(self, task: str) -> None:
        """Run the workflow optimization for the given task."""
        print(f"🚀 Starting OpenAI-powered workflow optimization...")
        print(f"Task: {task}")
        print("=" * 70)
        
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": task}
        ]
        
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")
            
            # Get tools specification
            tools = self.tool_registry.get_tools_spec()
            
            # Call LLM
            response = self.call_llm(messages, tools)
            
            print(f"💭 Agent thought: {response.thought}")
            
            # Add assistant message to conversation
            messages.append({
                "role": "assistant",
                "content": response.thought
            })
            
            # Execute tool calls
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    result = self.execute_tool_call(tool_call)
                    
                    # Add tool result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": f"call_{tool_call.tool_name}",
                        "content": result
                    })
                    
                    print(f"🔧 Tool result: {result[:200]}...")
            else:
                print("✅ No more tool calls needed")
                break
        
        print(f"\n🏁 Workflow optimization complete!")
        print(f"Iterations: {iteration}/{max_iterations}")

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python inference.py <task>")
        print("Example: python inference.py 'optimize the build process'")
        sys.exit(1)
    
    task = " ".join(sys.argv[1:])
    
    agent = OpenEnvAgent()
    agent.run_workflow(task)

if __name__ == "__main__":
    main()