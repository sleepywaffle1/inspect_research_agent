from uuid import uuid4

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import convert_to_messages
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from inspect_ai.agent import Agent, AgentState, agent, agent_bridge
from inspect_ai.model import messages_to_openai

import openai
from langchain_core.messages import BaseMessage
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.outputs import ChatResult
from typing import Any
from langchain_core.runnables.config import run_in_executor
from langchain_openai.chat_models.base import (
    _handle_openai_api_error,
    _handle_openai_bad_request,
)

# miniaml edit to ensure agent_bridge works with LangChain
class InspectBridgeChatOpenAI(ChatOpenAI):
    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._get_request_payload(messages, stop=stop, **kwargs)

        generation_info = None
        raw_response = None

        # error occurs in the else portion of original _agenerate codes
        try:
            raw_response = await self.async_client.with_raw_response.create(
                **payload
            )

            # Normal OpenAI raw response has .parse().
            # Inspect agent_bridge may already return ChatCompletion.
            if hasattr(raw_response, "parse"):
                response = raw_response.parse()
            else:
                response = raw_response

        except openai.BadRequestError as e:
            _handle_openai_bad_request(e)
        except openai.APIError as e:
            _handle_openai_api_error(e)
        except Exception as e:
            if raw_response is not None and hasattr(raw_response, "http_response"):
                e.response = raw_response.http_response
            raise e

        if (
            self.include_response_headers
            and raw_response is not None
            and hasattr(raw_response, "headers")
        ):
            generation_info = {"headers": dict(raw_response.headers)}

        return await run_in_executor(
            None,
            self._create_chat_result,
            response,
            generation_info,
        )

@agent
def web_research_agent(*, max_results: int = 5) -> Agent:
    """LangChain Tavili search agent.

    Args:
       max_results: Max search results to return (for use of Tavily search tool)

    Returns:
       Agent function for handling samples. May be passed to Inspect `bridge()`
       to create a standard Inspect solver.
    """

    # Sample handler
    async def execute(state: AgentState) -> AgentState:
        # Use bridge to map OpenAI Completions API to Inspect
        async with agent_bridge(state) as bridge:
            # Use OpenAI interface -- will be redirected to current Inspect model
            # by the agent_bridge() context manager.
            # model = ChatOpenAI(model="inspect")
            # model = init_chat_model(model="inspect")
            model = InspectBridgeChatOpenAI(
                model="inspect",
                api_key="dummy",
                base_url="http://localhost:13131/v1",
                include_response_headers=False,
            )

            # Configure web research tools/agent
            tools = [TavilySearchResults(max_results=max_results)]
            executor = create_react_agent(
                model=model,
                tools=tools,
                checkpointer=MemorySaver(),
            )

            # Read input (convert to LangChain)
            openai_messages = await messages_to_openai(state.messages)
            input = convert_to_messages(openai_messages)

            # Execute the agent
            await executor.ainvoke(
                input={"messages": input},
                config={"configurable": {"thread_id": uuid4()}},
            )

            # Return state from bridge
            return bridge.state

    return execute