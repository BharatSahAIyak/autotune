import json
import logging
import traceback

from django.conf import settings
from django.shortcuts import get_object_or_404
from jsonschema.exceptions import ValidationError
from jsonschema.validators import validate
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .models import Examples, WorkflowConfig
from .pydantic_models import QAPair, QAResponse

logger = logging.getLogger(__name__)


def call_llm_generate(prompt_text, workflow, config):
    llm_model = workflow.llm_model
    open_ai_key = settings.OPENAI_API_KEY
    parameters = {}
    if config.parameters:
        parameters = json.loads(config.parameters)
    llm = ChatOpenAI(
        model=llm_model,
        openai_api_key=open_ai_key,
        max_tokens=parameters.get("max_tokens", 2048),
        temperature=parameters.get("temperature", 0.7),
    )
    json_output_parser = PydanticOutputParser(pydantic_object=QAPair)
    format_instructions = json_output_parser.get_format_instructions()

    system_prompt = config.system_prompt

    system_prompt += f"\n\n{format_instructions}\n\n"

    system_prompt += "YOU MUST FOLLOW THE ABOVE SCHEMA IN THE RESPONSE. YOU WILL BE PENALIZED FOR NOT DOING SO."

    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt_text),
        ]
    )
    json_output_parser = StrOutputParser()

    chain = prompt_template | llm | json_output_parser

    try:
        parsed_response = chain.invoke(
            {"user_prompt": prompt_text, "system_prompt": system_prompt}
        )
        cleaned_data = parsed_response.strip("`json \n")
        parsed_response = json.loads(cleaned_data)
        validate_json(parsed_response, config.json_schema)

    except Exception as e:
        traceback.print_exc()
        parsed_response = []

    return parsed_response


def parse_and_save_examples(workflow, response):
    try:
        print(response)
        qa_response = QAResponse.parse_obj({"qa_pairs": response})
        for qa_pair in qa_response.qa_pairs:
            Examples.objects.create(
                workflow=workflow,
                text=json.dumps(
                    {"question": qa_pair.question, "answer": qa_pair.answer}
                ),
            )
    except Exception as e:
        print(f"Error parsing response: {str(e)}")
        raise


def generate_or_refine(workflow, refine=False):
    num_samples = settings.LLM_GENERATION_NUM_SAMPLES
    config = get_object_or_404(WorkflowConfig, name=workflow.workflow_type)
    prompt_text = construct_prompt(workflow, config, refine, num_samples)
    response = call_llm_generate(prompt_text, workflow, config)
    if response:
        parse_and_save_examples(workflow, response)
    return response


def construct_prompt(workflow, config, refine=False, num_samples=10):
    """
    Construct the user prompt to send to the language model based on the workflow settings,
    whether to refine based on existing examples, and the specified number of samples to generate.
    """
    source = workflow.prompt.source
    user_prompt = workflow.prompt.user
    user_prompt_template = config.user_prompt_template

    prompt = ""
    user_prompt = user_prompt_template.replace("{{.UserQuestion}}", user_prompt)

    if source:
        user_prompt = user_prompt.replace("{{.DocumentChunk}}", source)
    else:
        lines = user_prompt.split("\\n")
        filtered_lines = [
            line
            for line in lines
            if "{{.DocumentChunk}}" not in line and "Here is the document:" not in line
        ]
        user_prompt = "\n".join(filtered_lines)

    if refine:
        examples = workflow.examples.all()
        example_texts = ""
        for example in examples:
            example_text = json.loads(example.text)
            example_texts += f'\n{{"question": "{example_text["question"]}", "answer": "{example_text["answer"]}", "label": "{example.label}", "reason": "{example.reason}"}}'

        prompt += f"{user_prompt}\n\nBased on the examples below, refine and generate {num_samples} new examples.\n{example_texts}\n"
    else:
        post_text = f"\nPlease generate {num_samples} new examples based on the instructions given above."
        prompt += f"{user_prompt}{post_text}"

    return prompt


def validate_json(response, schema):
    try:
        validate(instance=response, schema=schema)
        print("JSON is valid against the schema.")
    except ValidationError as e:
        print("JSON is not valid. Reason:", e)
