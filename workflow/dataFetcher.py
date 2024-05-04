import json
import logging
from typing import List

from django.conf import settings
from django.db import transaction
from django.db.models import F, Q
from django.shortcuts import get_object_or_404
from gevent import joinall, spawn
from openai import OpenAI

from .models import Examples, Prompt, Task, WorkflowConfig, Workflows

logger = logging.getLogger(__name__)

open_ai_key = settings.OPENAI_API_KEY
client = OpenAI(api_key=open_ai_key)


class DataFetcher:
    def __init__(self, max_iterations, max_concurrent_fetches, batch_size) -> None:
        self.generated = 0
        self.examples = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.max_iterations = max_iterations
        self.max_concurrent_fetches = max_concurrent_fetches
        self.batch_size = batch_size

    def generate_or_refine(
        self,
        workflow_id,
        total_examples,
        workflow_config_id,
        llm_model,
        Model,
        refine=False,
        task_id=None,
        iteration=None,
    ):
        if iteration is not None and iteration > self.max_iterations:
            logger.error("Max iterations reached")
            return
        user_prompt, prompt_id = self.construct_user_prompt(workflow_id, refine)
        logger.info(workflow_config_id)
        config = get_object_or_404(WorkflowConfig, id=workflow_config_id)
        try:
            total_batches = max(
                1,
                (total_examples - self.generated + self.batch_size - 1)
                // self.batch_size,
            )

            greenlets = [
                spawn(
                    self.request_and_save,
                    user_prompt,
                    workflow_config_id,
                    llm_model,
                    iteration,
                    batch_index,
                    workflow_id,
                    task_id,
                    Model,
                    config.fields,
                    prompt_id,
                )
                for batch_index in range(
                    min(total_batches, self.max_concurrent_fetches)
                )
            ]

            joinall(greenlets)

            if task_id is not None:
                task = get_object_or_404(Task, id=task_id)
                task.generated_samples = self.generated
                logger.info(task.generated_samples)
                logger.info(task.updated_at)
                task.save()
                logger.info(task.updated_at)

            if self.generated < total_examples and iteration < self.max_iterations:
                self.generate_or_refine(
                    workflow_id=workflow_id,
                    total_examples=total_examples,
                    workflow_config_id=workflow_config_id,
                    llm_model=llm_model,
                    Model=Model,
                    refine=refine,
                    task_id=task_id,
                    iteration=iteration + 1,
                )
        except Exception as e:
            print(f"Error generating examples: {str(e)}")
            self.generate_or_refine(
                workflow_id=workflow_id,
                total_examples=total_examples,
                workflow_config_id=workflow_config_id,
                llm_model=llm_model,
                Model=Model,
                refine=True,
                task_id=task_id,
                iteration=iteration + 1,
            )

    def request_and_save(
        self,
        user_prompt,
        workflow_config_id,
        llm_model,
        iteration,
        batch_index,
        workflow_id,
        task_id,
        Model,
        fields,
        prompt_id,
    ):
        response = self.call_llm_generate(
            user_prompt, workflow_config_id, llm_model, iteration, batch_index
        )

        logger.info("response received from LLM")

        if response:
            self.generated += self.parse_and_save_examples(
                workflow_id=workflow_id,
                response=response,
                Model=Model,
                fields=fields,
                task_id=task_id,
                prompt_id=prompt_id,
            )

    def construct_user_prompt(self, workflow_id, refine=False):
        """
        Construct the user prompt to send to the language model based on the workflow settings,
        whether to refine based on existing examples, and the specified number of samples to generate.
        """
        workflow = Workflows.objects.get(workflow_id=workflow_id)
        config = get_object_or_404(WorkflowConfig, id=workflow.workflow_config.id)
        user_prompt_object: Prompt = workflow.latest_prompt
        user_prompt = user_prompt_object.user_prompt
        user_prompt_template = config.user_prompt_template

        prompt = ""
        user_prompt = user_prompt_template.replace(
            "{{workflow.user_prompt}}", user_prompt
        )

        if refine:
            examples: List[Examples] = workflow.examples.filter(
                Q(task_id__isnull=True)
                & ~Q(reason="")  # Adds condition for non-empty reason
            )
            example_texts = ""
            for example in examples:
                example_text = example.text
                dynamic_text = {key: example_text[key] for key in example_text}
                dynamic_text["label"] = example.label
                dynamic_text["reason"] = example.reason
                example_texts += f"\n{json.dumps(dynamic_text,indent=2)}"

            prompt += f"{user_prompt}\n\nBased on the examples below, refine and generate {self.batch_size} new examples.\n{example_texts}\n"
            prompt += f"Return the examples in a JSON object under an appropraite key following these pydantic models.\n\n {config.model_string}"
        else:
            post_text = f"\nPlease generate {self.batch_size} new examples based on the instructions given above."
            post_text += f"Return the examples in a JSON object under an appropraite key following these pydantic models.\n\n {config.model_string}"
            prompt += f"{user_prompt}{post_text}"

        return prompt, user_prompt_object.id

    def call_llm_generate(
        self, user_prompt, workflow_config_id, llm_model, iteration=None, batch=None
    ):
        logger.info(f"Running query for iteration {iteration} and batch {batch}")
        config = get_object_or_404(WorkflowConfig, id=workflow_config_id)

        system_prompt = config.system_prompt

        system_prompt += (
            "\nEnsure that the JSON output adheres to the provided structure\n"
        )
        system_prompt += "Pydantic classes for json structure: \n"

        system_prompt += f"\n{config.model_string}\n"

        logger.info(f"system_prompt: {system_prompt}")
        logger.info(f"user_prompt: {user_prompt}")
        if (
            llm_model == "gpt-4-turbo-preview"
            or llm_model == "gpt-4-turbo"
            or llm_model == "gpt-3.5-turbo-0125"
            or llm_model == "gpt-3.5-turbo"
        ):
            chat_completion = client.chat.completions.create(
                model=llm_model,
                temperature=config.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            response = chat_completion.choices[0].message.content
        else:
            chat_completion = client.chat.completions.create(
                model=llm_model,
                temperature=config.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            response = chat_completion.choices[0].message.content
            cleaned_data = response.strip("`json \n")
            response = json.loads(cleaned_data)

        self.input_tokens += chat_completion.usage.prompt_tokens
        self.output_tokens += chat_completion.usage.completion_tokens
        return response

    def parse_and_save_examples(
        self, workflow_id, response, Model, fields, prompt_id, task_id=None
    ):
        workflow = Workflows.objects.get(workflow_id=workflow_id)
        try:
            logger.info(response)
            response = json.loads(response)
            keys = list(response.keys())
            generated_examples = 0
            with transaction.atomic():
                for key in keys:
                    key_response = response[key]
                    data_values = list(key_response)
                    for data in data_values:
                        validated_data = Model.model_validate(data, strict=True)
                        example_data = validated_data.dict()
                        example = Examples.objects.create(
                            workflow=workflow,
                            text=example_data,
                            label=key,
                            task_id=task_id,
                            prompt_id=prompt_id,
                        )
                        if task_id is None:
                            self.examples.append(
                                {
                                    "example_id": str(example.example_id),
                                    "text": example.text,
                                    "label": example.label,
                                    "reason": "",
                                    "prompt_id": example.prompt.id,
                                }
                            )

                    logger.info(f"generated {len(data_values)} examples for key {key}")
                    generated_examples += len(data_values)

        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            raise
        return generated_examples
