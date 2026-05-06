from datetime import datetime

from inspect_ai.solver import solver, TaskState, Generate
from inspect_ai.agent import run
from inspect_ai.model import get_model, ChatMessageSystem, ChatMessageUser
from inspect_ai.tool import web_search

from agents.research_agent import research_agent
from prompts import (
    compress_research_system_prompt,
    compress_research_human_message,
    research_agent_prompt,
)
from tools.think_tool import think_tool

# ===== UTILITY FUNCTIONS =====

def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %-d, %Y")


@solver
def research_agent_solver(
    researcher_model: str = "openrouter/openai/gpt-4.1-mini",
    compress_model: str = "openrouter/openai/gpt-4.1-mini",
    mode: str = "full" # or terminate for stepwise evaluation
):
    async def solve(state: TaskState, generate: Generate) -> TaskState:

        if mode == "terminate":
            messages = [ChatMessageSystem(content=research_agent_prompt), *state.messages]

            model = get_model(researcher_model)

            output = await model.generate(
                messages,
                tools=[
                    web_search({"tavily": {"max_results": 5}}),
                    think_tool(),
                ],
            )

            tool_calls = output.message.tool_calls or []

            decision = "continue" if tool_calls else "stop"

            state.messages.append(output.message)
            state.metadata["next_step"] = decision
            state.output.completion = decision

            return state

        # full mode runs the entire agent end-to-end and output a compressed research
        research_topic = (
            state.metadata.get("research_brief") # from earlier research_brief_solver
            or state.output.completion
            or str(state.input)
        )

        # runs the react agent and gets the raw research notes
        agent_state = await run(
            research_agent(model=researcher_model),
            input=research_topic,
        )

        raw_research = agent_state.output.completion

        model = get_model(compress_model)

        compression = await model.generate(
            [
                ChatMessageSystem(
                    content=compress_research_system_prompt.format(
                        date=get_today_str()
                    )
                ),
                ChatMessageUser(
                    content=(
                        f"Research topic:\n{research_topic}\n\n"
                        f"Raw research notes:\n{raw_research}\n\n"
                        f"{compress_research_human_message.format(research_topic=research_topic)}"
                    )
                ),
            ]
        )

        state.metadata["research_topic"] = research_topic
        state.metadata["raw_notes"] = raw_research
        state.metadata["compressed_research"] = compression.completion

        state.output.completion = compression.completion

        return state

    return solve
