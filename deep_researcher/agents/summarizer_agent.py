"""
Agent used to summarize a research report based on a given prompt.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Union
from .baseclass import ResearchAgent
from ..llm_config import LLMConfig, model_supports_structured_output
from .utils.parse_output import create_type_parser

class SummarizerOutput(BaseModel):
    """Output from the Summarizer Agent"""
    question: str = Field(description="The original question that was researched", default="")
    answer: str = Field(description="The summarized answer to the question", default="")
    detail: Union[str, List[str]] = Field(description="Detailed explanation and context for the answer", default="")
    citations: List[str] = Field(description="List of citations used to support the answer", default_factory=list)

INSTRUCTIONS = """
You are a summarization expert. Your task is to read a research report and a prompt, and then generate a structured summary based on the content.

You will be given:
1. A research report.
2. A prompt that provides instructions on how to summarize the report.

Your task is to:
1. Carefully read the report and the prompt.
2. Generate a summary that follows the instructions in the prompt.
3. Format the output as a JSON object that adheres to the following schema.
- Make sure to escape any special characters (e.g. quotes) in the JSON output

Only output JSON. Follow the JSON schema below. Do not output anything else. I will be parsing this with Pydantic so output valid JSON only:
{SummarizerOutput.model_json_schema()}
"""

def init_summarizer_agent(config: LLMConfig) -> ResearchAgent:
    selected_model = config.fast_model

    return ResearchAgent(
        name="SummarizerAgent",
        instructions=INSTRUCTIONS,
        model=selected_model,
        output_type=SummarizerOutput if model_supports_structured_output(selected_model) else None,
        output_parser=create_type_parser(SummarizerOutput) if not model_supports_structured_output(selected_model) else None
    )