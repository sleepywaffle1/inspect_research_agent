from pathlib import Path
from datetime import datetime

from inspect_ai.agent import agent, react, Agent
from inspect_ai.tool import web_search, mcp_server_stdio

from prompts import research_agent_prompt
from tools.think_tool import think_tool

# ===== UTILITY FUNCTIONS =====
def get_current_dir() -> Path:
    """Get the current directory of the module.

    This function is compatible with Jupyter notebooks and regular Python scripts.

    Returns:
        Path object representing the current directory
    """
    try:
        return Path(__file__).resolve().parent
    except NameError:  # __file__ is not defined
        return Path.cwd()

def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %-d, %Y")


@agent
def research_agent(
    model: str = "openrouter/openai/gpt-4.1-mini",
) -> Agent:
    # # MCP server configuration for filesystem access
    # filesystem_server = mcp_server_stdio(
    #     name="filesystem",
    #     command="npx", 
    #     args=["-y", "@modelcontextprotocol/server-filesystem", str(get_current_dir() / "files")],  # Path to research documents
    # )

    return react(
        name="researcher",
        description="Performs iterative web research and submits research findings.",
        prompt=research_agent_prompt.format(date=get_today_str()),
        tools=[
            web_search({"tavily": {"max_results": 5}}),
            think_tool(),
        ],
        model=model,
        submit=True,
    )