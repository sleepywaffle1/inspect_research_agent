import asyncio
from pydantic import BaseModel, Field
from datetime import datetime

from inspect_ai.agent import run
from inspect_ai.solver import solver, TaskState, Generate
from inspect_ai.model import (
    get_model,
    ChatMessageSystem,
    ChatMessageUser,
    ChatMessageTool,
)
from inspect_ai.tool import tool

from prompts import lead_researcher_prompt
from tools.think_tool import think_tool
from solvers.research_agent_solver import research_agent_solver
from agents.research_agent import research_agent
from prompts import (
    compress_research_system_prompt,
    compress_research_human_message,
)

# ===== UTILITY FUNCTIONS =====

def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %-d, %Y")


class ConductResearch(BaseModel):
    """Tool for delegating a research task to a specialized sub-agent."""
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )

class ResearchComplete(BaseModel):
    """Tool for indicating that the research process is complete."""
    pass

# forces the supervisor_solver to reflect and come up with the next research topic(s) to delegate to the research_agent
@tool
def conduct_research():
    async def execute(research_topic: str) -> str:
        """
        Delegate a research task to a specialized sub-agent.

        Args:
            research_topic:The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).
        """
        # This tool is handled manually by the supervisor solver.
        return research_topic

    return execute


@tool
def research_complete():
    async def execute() -> str:
        """
        Tool for indicating that the research process is complete.
        """
        return "Research complete."

    return execute


@solver
def research_supervisor_solver(
    supervisor_model: str = "openrouter/openai/gpt-4.1-mini",
    researcher_model: str = "openrouter/openai/gpt-4.1-mini",
    compress_model: str = "openrouter/openai/gpt-4.1-mini",
    max_researcher_iterations: int = 6,
    max_concurrent_researchers: int = 3,
):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model = get_model(supervisor_model)

        # obtain research_brief from research_brief_solver
        research_brief = (
            state.metadata.get("research_brief")
            or state.output.completion
            or str(state.input)
        )

        # starts a new supervisor_messages chain
        supervisor_messages = [
            ChatMessageUser(content=research_brief)
        ]

        all_notes: list[str] = []
        all_raw_notes: list[str] = []

        # for each iteration, the supervisor can choose to delegate multiple research tasks to the research agent in parallel, or reflect using the think tool, or complete the research process
        system_message = ChatMessageSystem(
            content=lead_researcher_prompt.format(
                date=get_today_str(), 
                max_concurrent_research_units=max_concurrent_researchers,
                max_researcher_iterations=max_researcher_iterations
            )
        )
        for iteration in range(max_researcher_iterations):
            output = await model.generate(
                [system_message, *supervisor_messages],
                tools=[
                    conduct_research(),
                    research_complete(),
                    think_tool(),
                ],
            )

            supervisor_messages.append(output.message)

            tool_calls = output.message.tool_calls or []

            if not tool_calls:
                break

            if any(call.function == "research_complete" for call in tool_calls):
                break

            tool_messages = []

            think_calls = [
                call for call in tool_calls
                if call.function == "think_tool"
            ]

            conduct_calls = [
                call for call in tool_calls
                if call.function == "conduct_research"
            ]

            # Handle think_tool calls
            for call in think_calls:
                reflection = call.arguments.get("reflection", "")
                content = f"Reflection recorded: {reflection}"

                tool_messages.append(
                    ChatMessageTool(
                        content=content,
                        tool_call_id=call.id,
                    )
                )

            # Handle ConductResearch calls in parallel
            if conduct_calls:
                coros = []

                for call in conduct_calls:
                    research_topic = call.arguments["research_topic"]

                    async def run_one(topic: str):
                        # sub_state = TaskState(
                        #     input=[ChatMessageUser(content=topic)],
                        #     messages=[ChatMessageUser(content=topic)],
                        #     metadata={"research_brief": topic},
                        # )
                        # sub_solver = research_agent_solver(mode="full")
                        # return await sub_solver(sub_state, generate)

                        agent_state = await run(
                            research_agent(model=researcher_model),
                            input=topic,
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
                                        f"Research topic:\n{topic}\n\n"
                                        f"Raw research notes:\n{raw_research}\n\n"
                                        f"{compress_research_human_message}"
                                    )
                                ),
                            ]
                        )

                        return {
                            "compressed_research": compression.completion,
                            "raw_notes": raw_research,
                        }

                    coros.append(run_one(research_topic))

                results = await asyncio.gather(*coros)

                for call, result in zip(conduct_calls, results):
                    compressed = result.get(
                        "compressed_research",
                        ""
                    )

                    raw = result.get("raw_notes", "")

                    all_notes.append(compressed)
                    all_raw_notes.append(raw)

                    tool_messages.append(
                        ChatMessageTool(
                            content=compressed,
                            tool_call_id=call.id,
                        )
                    )

            supervisor_messages.extend(tool_messages)

        state.metadata["research_brief"] = research_brief
        state.metadata["supervisor_messages"] = supervisor_messages
        state.metadata["notes"] = all_notes
        state.metadata["raw_notes"] = all_raw_notes
        state.metadata["research_iterations"] = iteration + 1

        state.output.completion = "\n\n".join(all_notes)

        return state

    return solve