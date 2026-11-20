"""Replay the evidence tool contract through two independent offline agent frameworks."""

import json
from importlib.metadata import version

from agents import Agent as OpenAIAgent, Model as OpenAIModel, ModelResponse as OpenAIResponse
from agents import RunConfig, Runner, Usage, function_tool
from agno.agent import Agent as AgnoAgent
from agno.models.base import Model as AgnoModel
from agno.models.response import ModelResponse as AgnoResponse
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from smolagents import Model, ToolCallingAgent, tool
from smolagents.models import ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction, MessageRole

from workflow.local_assistant import lookup
from workflow.local_eval import CASES


def evidence(question: str) -> str:
    """Return the first verified source ID for a question, or no_evidence.

    Args:
        question: The evidence question, at most 160 characters.
    """
    records = lookup(question)
    return records[0]["id"] if records else "no_evidence"


def pydantic_replay(question: str) -> str:
    def respond(messages, _info):
        for part in messages[-1].parts:
            if isinstance(part, ToolReturnPart) and part.tool_name == "evidence":
                return ModelResponse(parts=[TextPart(part.content)])
        return ModelResponse(parts=[ToolCallPart("evidence", {"question": question})])

    agent = Agent(FunctionModel(respond), tools=[evidence], instrument=False)
    return agent.run_sync(question).data


class FixtureToolModel(Model):
    def __init__(self, question: str):
        super().__init__(model_id="offline-tool-contract")
        self.question = question

    def generate(self, messages, **kwargs):
        for message in reversed(messages):
            if message.role == MessageRole.TOOL_RESPONSE:
                if len(message.content) != 1 or "text" not in message.content[0]:
                    raise ValueError("Expected one text tool response")
                observed = message.content[0]["text"].removeprefix("Observation:\n").strip()
                return self.call("final_answer", {"answer": observed})
        return self.call("evidence", {"question": self.question})

    @staticmethod
    def call(name, arguments):
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            tool_calls=[ChatMessageToolCall(
                id=f"fixture-{name}", type="function",
                function=ChatMessageToolCallFunction(name=name, arguments=arguments),
            )],
        )


def smolagents_replay(question: str) -> str:
    agent = ToolCallingAgent(tools=[tool(evidence)], model=FixtureToolModel(question),
                             max_steps=2, verbosity_level=-1)
    return str(agent.run(question))


class OpenAIFixtureModel(OpenAIModel):
    def __init__(self, question: str):
        self.question = question

    async def get_response(self, system_instructions, input, *args, **kwargs):
        for item in reversed(input) if isinstance(input, list) else ():
            if isinstance(item, dict) and item.get("type") == "function_call_output":
                message = ResponseOutputMessage(
                    id="fixture-answer", status="completed", role="assistant", type="message",
                    content=[ResponseOutputText(type="output_text", text=item["output"], annotations=[])],
                )
                return OpenAIResponse([message], Usage(), None)
        call = ResponseFunctionToolCall(
            type="function_call", id="fixture-tool", call_id="evidence-1", name="evidence",
            arguments=json.dumps({"question": self.question}),
        )
        return OpenAIResponse([call], Usage(), None)

    def stream_response(self, *args, **kwargs):
        raise NotImplementedError("Only non-streaming fixture tests are supported")


def openai_replay(question: str) -> str:
    agent = OpenAIAgent(name="Offcut evidence replay", model=OpenAIFixtureModel(question),
                        tools=[function_tool(evidence)])
    result = Runner.run_sync(agent, question, max_turns=2, run_config=RunConfig(tracing_disabled=True))
    return result.final_output


class AgnoFixtureModel(AgnoModel):
    def invoke(self, messages, **kwargs):
        if messages[-1].role == "tool":
            return AgnoResponse(content=messages[-1].get_content_string())
        return AgnoResponse(tool_calls=[{
            "id": "fixture-evidence", "type": "function",
            "function": {"name": "evidence", "arguments": json.dumps({"question": messages[-1].content})},
        }])

    @staticmethod
    def parse_provider_response(response, **kwargs):
        return response

    def ainvoke(self, *args, **kwargs):
        raise NotImplementedError("This evaluator runs synchronously")

    def invoke_stream(self, *args, **kwargs):
        raise NotImplementedError("This evaluator does not stream")

    ainvoke_stream = invoke_stream
    parse_provider_response_delta = invoke_stream


def agno_replay(question: str) -> str:
    agent = AgnoAgent(model=AgnoFixtureModel(id="offline-tool-contract"), tools=[evidence],
                      telemetry=False, monitoring=False, tool_call_limit=1)
    return agent.run(question).content


def evaluate() -> dict:
    cases = [*CASES, ("astronomy galactic photon telescope", "no_evidence")]
    return {
        "mode": "Scripted model responses, real framework tool dispatch, no network inference",
        "frameworks": {
            distribution: {"version": version(distribution), "cases": [
                {"question": question, "expected": expected, "observed": replay(question)}
                for question, expected in cases
            ]}
            for distribution, replay in (
                ("pydantic-ai-slim", pydantic_replay), ("smolagents", smolagents_replay),
                ("openai-agents", openai_replay), ("agno", agno_replay),
            )
        },
    }


if __name__ == "__main__":
    report = evaluate()
    print(json.dumps(report, indent=2))
    if any(case["observed"] != case["expected"] for result in report["frameworks"].values()
           for case in result["cases"]):
        raise SystemExit(1)
