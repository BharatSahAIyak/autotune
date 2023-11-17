import asyncio
import hashlib
import json
import random
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models import GenerationAndCommitRequest
from tasks.data_fetcher import DataFetcher

from .fixtures import REDIS_DATA

MIN_WAIT_TIME = 0.01  # Minimum wait time in seconds
MAX_WAIT_TIME = 0.03  # Maximum wait time in seconds
CHANCE_OF_LESS_RESULTS = 0.8  # 80% chance of returning less than 20 results
MAX_RESULTS = 20  # Maximum number of results to return
MIN_RESULTS = 1  # Minimum number of results to return when not empty


GENERATION_AND_COMMIT_REQUEST = GenerationAndCommitRequest(
    prompt="test prompt",
    num_samples=100,
    repo="test_repo",
    split=[80, 10, 10],
    task="text_classification",
    num_labels=2,
    labels=["label1", "label2"],
    valid_data=None,
    invalid_data=None
)

redis_in_processing_mock_value = {
    "status": "Processing",
    "Progress": "80%",
    "data": json.dumps({"data": REDIS_DATA}),
    "Detail": "Generating data (Batch 0)",
}

REDIS_NO_DATA = {
    "data": json.dumps({"data": []}),
}


async def mock_get_data(system_content, api_key, task, labels, num_labels=None, valid_data=None, invalid_data=None):
    # Simulate a wait time
    await asyncio.sleep(random.uniform(MIN_WAIT_TIME, MAX_WAIT_TIME))

    # Decide the number of results to return
    if random.random() < CHANCE_OF_LESS_RESULTS:
        num_results = random.randint(MIN_RESULTS, MAX_RESULTS)
        selected_data = random.sample(REDIS_DATA, k=min(num_results, len(REDIS_DATA)))
        return [modify_data_with_hash(item) for item in selected_data]
    else:
        return []  # Return empty list


def modify_data_with_hash(data):
    # Create a unique hash for the text
    unique_hash = hashlib.md5(data["text"].encode()).hexdigest()
    modified_data = data.copy()
    modified_data["text"] = f"{unique_hash} :: {data['text']}"
    return modified_data


@pytest.mark.asyncio
async def test_initialization_from_redis():
    mock_redis = MagicMock()

    # We only care about the data key here
    mock_redis.hgetall = AsyncMock(return_value={"data": "[]"})
    with patch("utils.get_data", mock_get_data):
        fetcher = DataFetcher(
            GENERATION_AND_COMMIT_REQUEST, "openai_key", mock_redis, "task_id"
        )

        # pylint: disable=protected-access
        await fetcher._initialize_from_redis()
        assert fetcher.data == []
        assert fetcher.task_id == "task_id"
        assert fetcher.openai_key == "openai_key"

        # Test for non empty data
        mock_redis.hgetall = AsyncMock(return_value=redis_in_processing_mock_value)

        # pylint: disable=protected-access
        await fetcher._initialize_from_redis()
        assert len(fetcher.data["data"]) == 20


@pytest.mark.asyncio
async def test_fetch_and_update():
    mock_redis = MagicMock()
    mock_redis.hset = AsyncMock()
    mock_redis.hgetall = AsyncMock(return_value={})

    with patch("utils.get_data", mock_get_data):
        fetcher = DataFetcher(
            GENERATION_AND_COMMIT_REQUEST, "openai_key", mock_redis, "task_id"
        )
        assert len(fetcher.data["data"]) == 0
        await fetcher.fetch()

        # Assertions to check if data is fetched and Redis is updated
        assert len(fetcher.data["data"]) >= GENERATION_AND_COMMIT_REQUEST.num_samples
        assert fetcher.status["Progress"] == "100%"
        assert fetcher.status["data"] == json.dumps(fetcher.data)
        mock_redis.hset.assert_called()


# @pytest.mark.asyncio
# async def test_fetch_and_update_with_error():
#     with patch("utils.get_data", new=mock_get_data):
#         mock_redis = MagicMock()
#         mock_redis.hgetall = AsyncMock(return_value={})
#         mock_redis.hset = AsyncMock()

#         fetcher = DataFetcher(req, "openai_key", mock_redis, "task_id")
#         await fetcher._fetch_and_update(0)

#         # Check if the data is not extended when an exception occurs
#         assert len(fetcher.data["data"]) == 0


# @pytest.mark.asyncio
# async def test_redis_connection_error_on_initialization():
#     mock_redis = MagicMock()
#     mock_redis.hgetall = AsyncMock(
#         side_effect=ConnectionError("Redis connection error")
#     )

#     with patch("utils.get_data", new=mock_get_data):
#         fetcher = DataFetcher(req, "openai_key", mock_redis, "task_id")
#         await fetcher._initialize_from_redis()

#         # Assert that data is not modified on Redis connection error
#         assert fetcher.data == {"data": []}


# @pytest.mark.asyncio
# async def test_redis_timeout_during_data_fetching():
#     mock_redis = MagicMock()
#     mock_redis.hgetall = AsyncMock(return_value={})
#     mock_redis.hset = AsyncMock(side_effect=TimeoutError("Redis timeout error"))

#     with patch("utils.get_data", new=mock_get_data):
#         fetcher = DataFetcher(req, "openai_key", mock_redis, "task_id")
#         await fetcher._fetch_and_update(0)

#         # Assert the desired behavior on Redis timeout (e.g., logging an error)


# @pytest.mark.asyncio
# async def test_redis_general_exception():
#     mock_redis = MagicMock()
#     mock_redis.hgetall = AsyncMock(side_effect=Exception("General Redis exception"))
#     mock_redis.hset = AsyncMock(side_effect=Exception("General Redis exception"))

#     with pytest.raises(Exception("General Redis exception")):
#         with patch("utils.get_data", new=mock_get_data):
#             fetcher = DataFetcher(req, "openai_key", mock_redis, "task_id")
#             await fetcher._initialize_from_redis()

#         # Assert that the class handles a general Redis exception appropriately
