from datetime import datetime
from textwrap import dedent
from inspect_ai import Task, task
from inspect_ai.agent import react, deepagent
from inspect_ai.dataset import Sample
from inspect_ai.scorer import model_graded_qa
from inspect_ai.tool import web_search

from tools.think_tool import think_tool
from tools.clarify_tool import clarify_tool

from tools.generate_report_tool import generate_report_tool
from prompts import research_agent_prompt_with_additional_tools

# REPORT_JUDGE_TEMPLATE = """
# "You are an expert research analyst tasked with evaluating the quality of a research report based on the research question and the provided research brief.
# <Research Question>
# {question}
# </Research Question>

# <Research Brief>
# {brief}
# </Research Brief>

# <Web Search Results>
# {search_results}
# </Web Search Results>

# <Research Report>
# {report}
# </Research Report>

# Your evaluation should focus on the following criteria:
# 1. Relevance: Does the report directly address the research question and utilize the information from the web search results effectively?
# 2. Accuracy: Are the facts and information presented in the report correct and well-supported by the web search results?
# 3. Depth of Analysis: Does the report provide a thorough analysis of the topic, including insights and critical thinking, rather than just summarizing information?
# 4. Clarity and Coherence: Is the report well-organized, clearly written, and easy to understand?

# Output a score from 1 to 10 for each criterion, along with a brief justification for each score. 
# Then provide an overall assessment of the report's quality and any recommendations for improvement.
# """

# ===== UTILITY FUNCTIONS =====
def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %-d, %Y")

REPORT_JUDGE_CRITERIA = """
Your evaluation should focus on the following criteria:
1. Relevance: Does the report directly address the research question and utilize the information from the web search results effectively?
2. Accuracy: Are the facts and information presented in the report correct and well-supported by the web search results?
3. Depth of Analysis: Does the report provide a thorough analysis of the topic, including insights and critical thinking, rather than just summarizing information?
4. Clarity and Coherence: Is the report well-organized, clearly written, and easy to understand?
"""

@task
def inspect_react_agent_task():
    return Task(
        dataset=[
            Sample(
                input="What are the top coffee shops in San Francisco based on coffee quality? Focus on coffee beans quality and awards.",
                target=REPORT_JUDGE_CRITERIA
            ),
            Sample(
                input="Compare Tesla vs BYD electric vehicles in 2025. Focus on technical specifications, performance, and market reception.",
                target=REPORT_JUDGE_CRITERIA
            ),
        ],
        solver=react(
            prompt=research_agent_prompt_with_additional_tools.format(date=get_today_str()),
            todo_write=False,
            tools=[web_search({"tavily": {"max_results": 5}}), think_tool(), clarify_tool(), generate_report_tool()],
        ),
        scorer=model_graded_qa(model="openrouter/openai/gpt-4.1-mini"),
    )