from inspect_ai.tool import tool, Tool
from inspect_ai.model import (
    get_model,
    ChatMessageUser,
)

@tool
def generate_report_tool() -> Tool:
    async def execute(research_brief: str, findings: str) -> str:
        """Tool to generate a research report based on the research brief and findings.

        Use this tool after gathering sufficient information and insights to compile them into a coherent report.

        When to use:
        - After gathering information: When you have collected enough data, insights, and analysis from your research process and are ready to synthesize them into a final report.

        The generated report should include:
        1. An introduction that outlines the research question and the scope of the report.
        2. A summary of the key findings from the research, organized in a logical manner.
        3. An analysis section that provides insights, interpretations, and critical thinking based on the findings.
        4. A conclusion that summarizes the overall insights and may provide recommendations or implications based on the research.

        Args:
            research_brief: A summary of the research question and objectives.
            findings: A compilation of the key insights, data points, and analysis gathered during the research process.

        Returns:
            A well-structured research report that synthesizes the provided brief and findings.
        """

        model = get_model("openrouter/openai/gpt-4.1-mini")

        final_report_prompt = f"""
        Based on the following research brief and findings, generate a comprehensive research report.

        Research Brief:
        {research_brief}

        Findings:
        {findings}

        The report should include an introduction, a summary of key findings, an analysis section with insights, and a conclusion with overall assessment and recommendations.
        """

        final_report = await model.generate([ChatMessageUser(content=final_report_prompt)])

        return "Here is the final report: " + final_report.completion

    return execute