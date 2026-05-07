from inspect_ai.tool import tool, Tool
from inspect_ai.util import input_screen

@tool
def clarify_tool() -> Tool:
    async def execute(question: str) -> str:
        """Tool to obtain user's clarification if needed to resolve research gaps.

        Use this tool after receiving user's question if you assess you need to ask a clarifying question.

        When to use:
        - After receiving user's question: asess if the user's question is clear and specific enough to proceed with research?
        - When assessing research gaps: What specific information am I still missing?

        Clarification question should address:
        1. Specificity - What exactly are you looking for?
        2. Context - What is the background or context of your question?
        3. Scope - What is the extent or boundaries of your inquiry?
        4. Purpose - What is the intended use or outcome of your research?

        Args:
            question: the clarification question to ask the user

        Returns:
            User's response to the clarification question.
        """

        with input_screen() as console:
            console.print(f"Clarification needed: {question}")
            input = console.input("Please enter your response: ")

        return f"Clarification question recorded: {question}\n\nUser's response: {input}"

    return execute
