import json
import logging
from typing import List

from django.conf import settings
from gevent import joinall, spawn
from openai import OpenAI

from workflow.generator.dataFetcher import DataFetcher
from workflow.models import MLModel, MLModelConfig

logger = logging.getLogger(__name__)
open_ai_key = settings.OPENAI_API_KEY
client = OpenAI(api_key=open_ai_key)


class ModelDataFetcher:
    def __init__(self, model_config, model, pydantic_model) -> None:
        self.generated = 0
        self.examples = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.max_iterations = 5
        self.max_concurrent_fetches = 5
        self.batch_size = 10
        self.model_config: MLModelConfig = model_config
        self.model: MLModel = model
        self.pydantic_model = pydantic_model

    def generate_or_refine(
        self,
        input,
        output,
        task_type,
        iteration=0,
        total_examples=10,
    ):
        if iteration is not None and iteration > self.max_iterations:
            logger.error("Max iterations reached")
            return

        user_prompt = self.construct_user_prompt(input, output, total_examples)

        try:
            total_batches = max(
                1,
                (total_examples - self.generated + self.batch_size - 1)
                // self.batch_size,
            )

            print("this is total_batches", total_batches)

            greenlets = [
                spawn(self.request_and_save, user_prompt)
                for batch_index in range(
                    min(total_batches, self.max_concurrent_fetches)
                )
            ]

            joinall(greenlets)

            print("this is self.generated ", self.generated)

            if self.generated < total_examples and iteration < self.max_iterations:
                self.generate_or_refine(
                    input=input,
                    output=output,
                    task_type=task_type,
                    iteration=iteration + 1,
                )
        except Exception as e:
            print(f"Error generating examples: {str(e)}")
            self.generate_or_refine(
                input=input,
                output=output,
                task_type=task_type,
                iteration=iteration + 1,
            )

    def construct_user_prompt(self, input, output, total_examples):
        """
        Construct the user prompt to send to the language model based on the workflow settings,
        whether to refine based on existing examples, and the specified number of samples to generate.
        """

        user_prompt_template = self.model_config.user_prompt_template

        if self.model.task == "text_classification":
            choices = self.model.label_studio_element["config"]["choices"]
            wrong_choices = [choice for choice in choices if choice != output]
            user_prompt_template = user_prompt_template.replace(
                "{{target_class}}", output
            )
            user_prompt_template = user_prompt_template.replace("{{sentence}}", input)
            user_prompt_template = user_prompt_template.replace(
                "{{classes_other_than_target}}", ", ".join(wrong_choices)
            )

            user_prompt_template += f"Generate {total_examples} new      examples.Return the examples in a JSON object under an appropriate key following these pydantic models.\n\n {self.model_config.model_string}"

        return user_prompt_template

    def request_and_save(
        self,
        user_prompt,
    ):
        fetcher = DataFetcher(0, 0, 0)

        response = fetcher.call_llm_generate(
            user_prompt=user_prompt,
            system_prompt=self.model_config.system_prompt,
            model_string=self.model_config.model_string,
            temperature=1,
            llm_model="gpt-3.5-turbo-0125",
        )

        logger.info("response received from LLM")

        if response:
            self.generated += self.parse_examples(
                response=response,
            )

    def parse_examples(self, response):
        logger.info(response)
        response = json.loads(response)
        keys = list(response.keys())
        generated_examples = 0
        for key in keys:
            key_response = response[key]
            data_values = list(key_response)
            for data in data_values:
                validated_data = self.pydantic_model.model_validate(data, strict=True)
                example_data = validated_data.dict()

                self.examples.append(example_data)

            logger.info(f"generated {len(data_values)} examples for key {key}")
            generated_examples += len(data_values)

        return generated_examples
