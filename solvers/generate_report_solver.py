from datetime import datetime

from inspect_ai.solver import solver, TaskState, Generate
from inspect_ai.model import (
    get_model,
    ChatMessageUser,
)
from prompts import final_report_generation_prompt

# ===== UTILITY FUNCTIONS =====

def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %-d, %Y")

@solver
def generate_report_solver(
    model_name: str = "openrouter/openai/gpt-4.1-mini",
):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model = get_model(model_name)

        notes = state.metadata.get("notes", [])
        findings = "\n".join(notes)

        final_report_prompt = final_report_generation_prompt.format(
            research_brief=state.metadata.get("research_brief", ""),
            findings=findings,
            date=get_today_str()
        )

        final_report = await model.generate([ChatMessageUser(content=final_report_prompt)])

        state.output.completion = final_report.completion
        state.metadata["final_report"] = "Here is the final report: " + final_report.completion

        return state

    return solve
