from datetime import datetime
import json
from pydantic import BaseModel, Field, ValidationError

from inspect_ai.model import get_model, ChatMessageUser, ChatMessageAssistant, GenerateConfig, ResponseSchema
from inspect_ai.util import json_schema
from inspect_ai.solver import solver, TaskState, Generate

from prompts import (
    clarify_with_user_instructions,
    transform_messages_into_research_topic_prompt,
)

# ===== STRUCTURED OUTPUT SCHEMAS =====

class ClarifyWithUser(BaseModel):
    """Schema for user clarification decision and questions."""

    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )

class ResearchQuestion(BaseModel):
    """Schema for structured research brief generation."""

    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )

# ===== UTILITY FUNCTIONS =====

def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %-d, %Y")

def messages_to_text(state: TaskState) -> str:
    return "\n".join(
        f"{msg.role}: {msg.text}"
        for msg in state.messages
        if hasattr(msg, "text") and msg.text
    )


@solver
def pre_scope_solver(
    model_name: str = "openrouter/openai/gpt-4.1-mini",
):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model = get_model(model_name)

        conversation = messages_to_text(state)

        # 1. Clarification step
        clarify_prompt = clarify_with_user_instructions.format(
            messages=conversation,
            date=get_today_str(),
        )

        clarify_output = await model.generate(
            [ChatMessageUser(content=clarify_prompt)],
            config=GenerateConfig(
                temperature=0.0,
                response_schema=ResponseSchema(name="clarify_with_user", json_schema=json_schema(ClarifyWithUser), strict=True)
            ),
        )

        clarify = ClarifyWithUser.model_validate_json(clarify_output.completion)

        if clarify.need_clarification:
            state.messages.append(ChatMessageAssistant(content=clarify.question))
            state.output.completion = clarify.question
            state.metadata["needs_clarification"] = True
            return state

        state.messages.append(ChatMessageAssistant(content=clarify.verification))
        state.metadata["needs_clarification"] = False

        # 2. Research brief step
        updated_conversation = messages_to_text(state)

        brief_prompt = transform_messages_into_research_topic_prompt.format(
            messages=updated_conversation,
            date=get_today_str(),
        )

        brief_output = await model.generate(
            [ChatMessageUser(content=brief_prompt)],
            config=GenerateConfig(
                temperature=0.0,
                response_schema=ResponseSchema(name="research_question", json_schema=json_schema(ResearchQuestion), strict=True)
            ),
        )

        try:
            research_question = ResearchQuestion.model_validate_json(
                brief_output.completion
            )
        except ValidationError:
            research_question = ResearchQuestion(
                research_brief=brief_output.completion
            )

        state.metadata["research_brief"] = research_question.research_brief

        # This is equivalent to your LangGraph supervisor_messages handoff
        state.messages.append(
            ChatMessageUser(content=f"{research_question.research_brief}.")
        )

        state.output.completion = research_question.research_brief

        print(state)

        return state

    return solve