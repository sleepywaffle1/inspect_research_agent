from inspect_ai.solver import TaskState
from inspect_ai.scorer import (
    scorer,
    Score,
    Target,
    accuracy,
    stderr,
)
from inspect_ai.model import (
    get_model,
    ChatMessageUser,
    GenerateConfig,
    ResponseSchema,
)
from inspect_ai.util import json_schema
from pydantic import BaseModel, Field, ValidationError

from prompts import BRIEF_CRITERIA_PROMPT


class Criteria(BaseModel):
    criteria_text: str = Field(
        description="The specific success criterion being evaluated."
    )
    reasoning: str = Field(
        description="Explanation of whether the criterion is captured."
    )
    is_captured: bool = Field(
        description="Whether the criterion is captured in the research brief."
    )


@scorer(metrics=[accuracy(), stderr()])
def success_criteria_scorer(
    model_name: str = "openrouter/openai/gpt-4.1-mini",
):
    async def score(state: TaskState, target: Target) -> Score:
        model = get_model(model_name)

        research_brief = state.output.completion

        # target.text is your "\n".join(criteria)
        criteria = [
            line.strip()
            for line in target.text.splitlines()
            if line.strip()
        ]

        individual_evaluations = []

        for criterion in criteria:
            prompt = BRIEF_CRITERIA_PROMPT.format(
                research_brief=research_brief,
                criterion=criterion,
            )

            result = await model.generate(
                [ChatMessageUser(content=prompt)],
                config=GenerateConfig(
                    temperature=0.0,
                    response_schema=ResponseSchema(
                        name="criteria",
                        json_schema=json_schema(Criteria),
                        strict=True # may fail via OpenRouter
                    ),
                ),
            )

            try:
                parsed = Criteria.model_validate_json(result.completion)
            except ValidationError:
                parsed = Criteria(
                    criteria_text=criterion,
                    reasoning=f"Failed to parse judge output: {result.completion}",
                    is_captured=False,
                )

            individual_evaluations.append(parsed)

        captured_count = sum(
            1 for item in individual_evaluations if item.is_captured
        )
        total_count = len(individual_evaluations)
        score_value = captured_count / total_count if total_count else 0.0

        return Score(
            value=score_value,
            explanation="\n\n".join(
                [
                    f"Criterion: {item.criteria_text}\n"
                    f"Captured: {item.is_captured}\n"
                    f"Reasoning: {item.reasoning}"
                    for item in individual_evaluations
                ]
            ),
            metadata={
                "individual_evaluations": [
                    {
                        "criteria": item.criteria_text,
                        "captured": item.is_captured,
                        "reasoning": item.reasoning,
                    }
                    for item in individual_evaluations
                ]
            },
        )

    return score
