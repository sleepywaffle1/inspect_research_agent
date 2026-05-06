# inspect_research_agent

## Details
This repo implements the deep research agent from langchain-ai/deep_research_from_scratch using InspectAI framework.

## Setup
To setup, first git clone InspectAI from: https://github.com/UKGovernmentBEIS/inspect_ai

### Structured output with OpenRouter
To ensure InspectAI is able to support structured output for OpenRouter, I have added the following edits to the cloned library:
```
src/inspect_ai/model/_providers/openrouter.py/OpenRouterAPI/completion_params
        ...
        # own edit: to enable structured output
        if config.response_schema is not None:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": config.response_schema.name,
                    "schema": config.response_schema.json_schema,
                },
            }

        return params
```

### uv commands
```
uv venv --python3.12
uv sync
```

### Running the evaluations
```
PYTHONPATH=. inspect eval tasks/<TASK_NAME>.py --model openrouter/openai/gpt-4.1-mini
```

## agent_bridge() for LangChain
Agents can be bridged into Inspect using ```agent_bridge()```, where Inspect will intercept the model calling. 

#### Flow
Inspect messages converted to LangChain messages -> LangGraph agent runs -> Inspect bridge intercepts and call cli model -> returns response to LangGraph -> LangGraph finishes -> returns bridge.state to Inspect  

This tells Inspect to use OpenRouter as provider and gpt-4.1-mini as the model. 
```
inspect eval task.py --model openrouter/openai/gpt-4.1-mini
```

Agent implementation
```
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
            # Enters the bridge context
            # Use OpenAI interface -- will be redirected to current Inspect model
            # LangChain thinks: call OpenAI model "inspect"
            # Inspect bridge intercepts: route this to the Inspect eval model
            model = InspectBridgeChatOpenAI(
                model="inspect",
                api_key="dummy",
                base_url="http://localhost:13131/v1",
                include_response_headers=False,
            )

            # Configure LangChain web research tools/agent
            tools = [TavilySearchResults(max_results=max_results)]
            executor = create_react_agent(
                model=model,
                tools=tools,
                checkpointer=MemorySaver(),
            )

            # Inspect messages are converted to LangChain messages
            openai_messages = await messages_to_openai(state.messages)
            input = convert_to_messages(openai_messages)

            # Execute the agent
            # During running of the agent, LangChain eventually calls InspectBridgeChatOpenAI which will be intercepted by Inspect
            # Routes to the actual CLI model: Inspect will return the response to be parsed by LangChain
            await executor.ainvoke(
                input={"messages": input},
                config={"configurable": {"thread_id": uuid4()}},
            )

            # Return state from bridge
            return bridge.state

    return execute
```
However, there is a mismatch in the response schema returned by Inspect bridge and expected by LangGraph. 

A child class InspectBridgeChatOpenAI (under ```langchain_agent/agent.py```) is created to resolve this.


## Comments
- solvers can be chained but chain(...) only supports sequential pipeline of solver steps
- beneficial for intermediate modular evaluation (1 module = 1 solver)
- OpenRouter provider does not seem to support Filesystem MCP Server (validation error for read_file)
- inspect view to view the logs (very helpful and easy to use)