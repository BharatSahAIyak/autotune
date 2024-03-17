import json
import traceback

from django.conf import settings
from django.shortcuts import get_object_or_404
from jsonschema.exceptions import ValidationError
from jsonschema.validators import validate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .models import Examples, WorkflowConfig
from .pydantic_models import QAResponse, QAPair


def call_llm_generate(prompt_text, workflow, config):
    llm_model = workflow.llm_model
    open_ai_key = settings.OPENAI_API_KEY
    llm = ChatOpenAI(model=llm_model, openai_api_key=open_ai_key, max_tokens=config.parameters.get("max_tokens", 2048),
                 temperature=config.parameters.get("temperature", 0.7))
    json_output_parser = PydanticOutputParser(pydantic_object=QAPair)
    format_instructions = json_output_parser.get_format_instructions()
    # prompt = PromptTemplate(
    #     template=f"{prompt_text}\n{format_instructions}\n",
    #     input_variables=["prompt_text"],
    #     partial_variables={"format_instructions": json_output_parser.get_format_instructions()},
    # )
    prompt = ChatPromptTemplate.from_template("{query}")
    json_output_parser = StrOutputParser()
    chain = prompt | llm | json_output_parser

    try:
        parsed_response = chain.invoke({"query": prompt_text+"\n"+format_instructions})
        print(parsed_response)
        cleaned_data = parsed_response.strip('`json \n')
        print("\n")
        parsed_response = json.loads(cleaned_data)
        validate_json(parsed_response, config.json_schema)
    except Exception as e:
        traceback.print_exc()
        parsed_response = []

    return parsed_response


def parse_and_save_examples(workflow, response):
    try:
        print(response)
        qa_response = QAResponse.parse_obj({'qa_pairs': response})
        for qa_pair in qa_response.qa_pairs:
            Examples.objects.create(workflow=workflow,
                                    text=json.dumps({"question": qa_pair.question, "answer": qa_pair.answer}))
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
    Construct the prompt to send to the language model based on the workflow settings,
    whether to refine based on existing examples, and the specified number of samples to generate.
    """
    source = workflow.prompt.source
    user_prompt = workflow.prompt.user
    system_prompt = config.system_prompt
    user_prompt_template = config.user_prompt_template

    prompt = f"{system_prompt}\n"
    user_prompt = user_prompt_template.replace("{{.UserQuestion}}", user_prompt)

    if source:
        user_prompt = user_prompt.replace("{{.DocumentChunk}}", source)
    else:
        # user_prompt = user_prompt.replace("{{.DocumentChunk}}", '')
        # user_prompt = user_prompt.replace("Here is the document:", '')
        lines = user_prompt.split("\\n")
        filtered_lines = [line for line in lines if "{{.DocumentChunk}}" not in line and "Here is the document:" not in line]
        user_prompt = '\n'.join(filtered_lines)

    if refine:
        examples = workflow.examples.all()
        example_texts = "\n".join([
            f'{{"question": "{json.loads(e.text)["question"]}", "answer": "{json.loads(e.text)["answer"]}", "label": "{e.label}", "reason": "{e.reason}"}}'
            for e in examples
        ])
        # user_prompt = user_prompt.replace("{{.DocumentChunk}}", example_texts)
        prompt += f"{user_prompt}\nBased on the examples below, refine and generate {num_samples} new examples.{example_texts}\n"
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
