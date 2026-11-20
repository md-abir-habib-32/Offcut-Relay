"""Replay the evidence tool contract through two independent offline agent frameworks."""

import json
from importlib.metadata import version

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


def evaluate() -> dict:
    cases = [*CASES, ("astronomy galactic photon telescope", "no_evidence")]
    return {
        "mode": "Scripted model responses, real framework tool dispatch, no network inference",
        "frameworks": {
            distribution: {"version": version(distribution), "cases": [
                {"question": question, "expected": expected, "observed": replay(question)}
                for question, expected in cases
            ]}
            for distribution, replay in (("pydantic-ai-slim", pydantic_replay), ("smolagents", smolagents_replay))
        },
    }


if __name__ == "__main__":
    report = evaluate()
    print(json.dumps(report, indent=2))
    if any(case["observed"] != case["expected"] for result in report["frameworks"].values()
           for case in result["cases"]):
        raise SystemExit(1)
