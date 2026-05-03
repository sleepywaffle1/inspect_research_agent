# inspect_research_agent

## Details
This repo implements the deep research agent from langchain-ai/deep_research_from_scratch using InspectAI framework.

## Note
To ensure InspectAI is able to support structured output for OpenRouter, I have added the following edits to the library:
```
src/inspect_ai/model/_providers/openrouter.py/OpenRouterAPI/completion_params (line 272 onwards)
        # own edit: to enable structured output
        if config.response_schema is not None:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": config.response_schema.name,
                    "schema": config.response_schema.json_schema,
                },
            }
```

## Comments
- solvers can be chained but chain(...) only supports sequential pipeline of solver steps
- beneficial for intermediate modular evaluation (1 module = 1 solver)
- does not seem to support Filesystem MCP Server (validation error for read_file)