import json
import unittest
from unittest.mock import MagicMock, call, patch

import pytest

from workflow.generator.dataFetcher import DataFetcher
from workflow.models import Examples  # Add import for Examples model


@pytest.mark.django_db
class TestDataFetcher(unittest.TestCase):

    def setUp(self):
        self.max_iterations = 2
        self.batch_size = 2
        self.max_concurrent_fetches = 10

        self.dataFetcher = DataFetcher(
            max_iterations=self.max_iterations,
            batch_size=self.batch_size,
            max_concurrent_fetches=self.max_concurrent_fetches,
        )
        self.dataFetcher.input_tokens = 0
        self.dataFetcher.output_tokens = 0
        self.dataFetcher.generated = 0
        self.dataFetcher.examples = []

    def mock_workflow_config(self):
        return MagicMock(
            id="8101b811-046f-45f6-8215-f2fde3ba34f4",
            name="Pest Chunks!",
            system_prompt="You are a helpful data generation assistant working as a teacher. You are an expert in this field. Don't Hallucinate",
            user_prompt_template="{{workflow.user_prompt}}",
            schema_example={
                "text": "string",
                "choices": [
                    {"text": "string", "score": 0},
                    {"text": "string", "score": 0},
                    {"text": "string", "score": 1},
                    {"text": "string", "score": 0},
                ],
            },
            temperature=2,
            fields=[{"text": "str"}, {"choices": "List"}],
            model_string="class Choice(BaseModel):\n  text: 'str'\n  score: 'int'\n\nclass Model(BaseModel):\n  text: 'str'\n  choices: 'List[Choice]'",
        )

    def mock_workflow(self):
        workflow = MagicMock()
        workflow.id = 1
        workflow.workflow_config = MagicMock(id=1)
        return workflow

    @patch("workflow.generator.dataFetcher.get_object_or_404")
    def test_call_llm_generate(self, mock_get_object_or_404):
        # Mock the WorkflowConfig object
        mock_get_object_or_404.return_value = self.mock_workflow_config()

        # Call the method
        response = self.dataFetcher.call_llm_generate(
            user_prompt="test prompt",
            workflow_config_id=1,
            llm_model="gpt-3.5-turbo",
            iteration=1,
            batch=1,
        )

        # Check if the response is correct
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    @patch("workflow.generator.dataFetcher.Workflows.objects.get")
    @patch("workflow.generator.dataFetcher.Examples.objects.create")
    def test_parse_and_save_examples(self, mock_examples_create, mock_workflows_get):
        # Mock the Workflow object
        mock_workflow = self.mock_workflow()
        mock_workflows_get.return_value = mock_workflow

        # Mock the Model
        Model = MagicMock()

        # Update the mock to return the actual data being validated
        def mock_model_validate(data, strict):
            validated = MagicMock()
            validated.dict.return_value = data
            return validated

        Model.model_validate.side_effect = mock_model_validate

        # Capture the calls to Examples.objects.create
        mock_examples_create.return_value = MagicMock()

        # Call the method
        response_data = {
            "addition": [
                {
                    "text": "question 1",
                    "choices": [
                        {"text": "answer 1", "score": 0},
                        {"text": "answer 2", "score": 0},
                        {"text": "answer 3", "score": 1},
                        {"text": "answer 4", "score": 0},
                    ],
                },
                {
                    "text": "question 2",
                    "choices": [
                        {"text": "answer 1", "score": 0},
                        {"text": "answer 2", "score": 0},
                        {"text": "answer 3", "score": 1},
                        {"text": "answer 4", "score": 0},
                    ],
                },
            ],
            "subtraction": [
                {
                    "text": "question 3",
                    "choices": [
                        {"text": "answer 1", "score": 0},
                        {"text": "answer 2", "score": 0},
                        {"text": "answer 3", "score": 1},
                        {"text": "answer 4", "score": 0},
                    ],
                },
                {
                    "text": "question 4",
                    "choices": [
                        {"text": "answer 1", "score": 0},
                        {"text": "answer 2", "score": 0},
                        {"text": "answer 3", "score": 1},
                        {"text": "answer 4", "score": 0},
                    ],
                },
            ],
        }
        generated = self.dataFetcher.parse_and_save_examples(
            workflow_id=1,
            response=json.dumps(response_data),
            Model=Model,
            fields={"field1": "value1"},
            prompt_id=1,
            task_id=1,
        )

        # Verify that Examples.objects.create was called with the correct parameters
        calls = [
            call(
                workflow=mock_workflow,
                text=response_data["addition"][0],
                label="addition",
                task_id=1,
                prompt_id=1,
            ),
            call(
                workflow=mock_workflow,
                text=response_data["addition"][1],
                label="addition",
                task_id=1,
                prompt_id=1,
            ),
            call(
                workflow=mock_workflow,
                text=response_data["subtraction"][0],
                label="subtraction",
                task_id=1,
                prompt_id=1,
            ),
            call(
                workflow=mock_workflow,
                text=response_data["subtraction"][1],
                label="subtraction",
                task_id=1,
                prompt_id=1,
            ),
        ]
        mock_examples_create.assert_has_calls(calls, any_order=True)

        # Check if the examples are parsed and saved correctly
        self.assertEqual(generated, 4)
        self.assertEqual(len(self.dataFetcher.examples), 0)


if __name__ == "__main__":
    unittest.main()
