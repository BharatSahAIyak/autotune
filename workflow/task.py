import json
import logging
import traceback

from django.conf import settings
from django.shortcuts import get_object_or_404
from gevent import joinall, spawn
from jsonschema.exceptions import ValidationError
from jsonschema.validators import validate
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .models import Examples, WorkflowConfig, Workflows
from .pydantic_models import QAPair, QAResponse

logger = logging.getLogger(__name__)


batch_size = int(getattr(settings, "MAX_BATCH_SIZE", 10))
max_iterations = int(getattr(settings, "MAX_ITERATIONS", 100))


class DataFetcher:
    def __init__(self) -> None:
        self.generated = 0

    def generate_or_refine(
        self,
        workflow_id,
        total_examples,
        workflow_type,
        llm_model,
        refine=False,
        task_id=None,
        iteration=None,
    ):
        if iteration is not None and iteration > max_iterations:
            logger.error("Max iterations reached")
            return
        user_prompt = self.construct_user_prompt(workflow_id, refine)
        config = get_object_or_404(WorkflowConfig, name=workflow_type)
        if task_id is not None:
            try:
                total_batches = max(
                    1,
                    (total_examples - self.generated + batch_size - 1) // batch_size,
                )

                greenlets = [
                    spawn(
                        self.request_and_save,
                        user_prompt,
                        workflow_type,
                        llm_model,
                        iteration,
                        batch_index,
                        config.json_schema,
                        workflow_id,
                        task_id,
                    )
                    for batch_index in range(total_batches)
                ]

                joinall(greenlets)

                if self.generated < total_examples and iteration < max_iterations:
                    self.generate_or_refine(
                        workflow_id=workflow_id,
                        total_examples=total_examples,
                        workflow_type=workflow_type,
                        llm_model=llm_model,
                        refine=refine,
                        task_id=task_id,
                        iteration=iteration + 1,
                        batch=0,
                    )
            except Exception as e:
                print(f"Error generating examples: {str(e)}")
                self.generate_or_refine(
                    workflow_id=workflow_id,
                    total_examples=total_examples,
                    workflow_type=workflow_type,
                    llm_model=llm_model,
                    refine=True,
                    task_id=task_id,
                    iteration=iteration + 1,
                )
        else:
            try:
                response = self.call_llm_generate(user_prompt, workflow_type, llm_model)
                if response:
                    cleaned_data = response.strip("`json \n")
                    parsed_response = json.loads(cleaned_data)
                    self.validate_json(parsed_response, config.json_schema)
                    self.parse_and_save_examples(workflow_id, parsed_response)
                    return parsed_response
            except Exception as e:
                print(f"Error generating examples: {str(e)}")
                # self.generate_or_refine(
                #     workflow_id=workflow_id,
                #     total_examples=total_examples,
                #     workflow_type=workflow_type,
                #     llm_model=llm_model,
                #     refine=refine,
                # )
                return str(e)

    def request_and_save(
        self,
        user_prompt,
        workflow_type,
        llm_model,
        iteration,
        batch_index,
        json_schema,
        workflow_id,
        task_id,
    ):
        response = self.call_llm_generate(
            user_prompt, workflow_type, llm_model, iteration, batch_index
        )

        if response:
            cleaned_data = response.strip("`json \n")
            parsed_response = json.loads(cleaned_data)
            self.validate_json(parsed_response, json_schema)
            if parsed_response:
                self.generated += self.parse_and_save_examples(
                    workflow_id, parsed_response, task_id
                )

    def construct_user_prompt(self, workflow_id, refine=False):
        """
        Construct the user prompt to send to the language model based on the workflow settings,
        whether to refine based on existing examples, and the specified number of samples to generate.
        """
        workflow = Workflows.objects.get(workflow_id=workflow_id)
        config = get_object_or_404(WorkflowConfig, name=workflow.workflow_type)
        num_samples = int(settings.LLM_GENERATION_NUM_SAMPLES) | 10
        user_prompt = workflow.prompt.user
        user_prompt_template = config.user_prompt_template

        prompt = ""
        user_prompt = user_prompt_template.replace("{{.UserQuestion}}", user_prompt)

        if refine:
            examples = workflow.examples.filter(task_id__isnull=True)
            example_texts = ""
            for example in examples:
                example_text = json.loads(example.text)
                example_texts += f'\n{{"question": "{example_text["question"]}", "answer": "{example_text["answer"]}", "label": "{example.label}", "reason": "{example.reason}"}}'

            prompt += f"{user_prompt}\n\nBased on the examples below, refine and generate {num_samples} new examples.\n{example_texts}\n"
        else:
            post_text = f"\nPlease generate {num_samples} new examples based on the instructions given above."
            prompt += f"{user_prompt}{post_text}"

        return prompt

    def call_llm_generate(
        self, user_prompt, workflow_type, llm_model, iteration=None, batch=None
    ):
        logger.info(f"Running query for iteration {iteration} and batch {batch}")
        config = get_object_or_404(WorkflowConfig, name=workflow_type)
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
                HumanMessage(content=user_prompt),
            ]
        )
        json_output_parser = StrOutputParser()

        chain = prompt_template | llm | json_output_parser

        try:
            return chain.invoke(
                {"user_prompt": user_prompt, "system_prompt": system_prompt}
            )

        except Exception as e:
            traceback.print_exc()
            parsed_response = []

    def parse_and_save_examples(self, workflow_id, response, task_id=None):
        workflow = Workflows.objects.get(workflow_id=workflow_id)
        try:
            qa_response = QAResponse.parse_obj({"qa_pairs": response})
            num_pairs = len(qa_response.qa_pairs)
            for qa_pair in qa_response.qa_pairs:
                Examples.objects.create(
                    workflow=workflow,
                    text=json.dumps(
                        {"question": qa_pair.question, "answer": qa_pair.answer}
                    ),
                    task_id=task_id,
                )
            logger.info(f"generated {num_pairs} examples")
            return num_pairs
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            raise

    def validate_json(self, response, schema):
        try:
            validate(instance=response, schema=schema)
            print("JSON is valid against the schema.")
        except ValidationError as e:
            print("JSON is not valid. Reason:", e)
