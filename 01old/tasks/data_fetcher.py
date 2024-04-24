import asyncio
import json
import logging

import coloredlogs

# from utils import get_data
from old import utils
from old.models.data import GenerationAndCommitRequest, QuestionCreationRequest

logger = logging.getLogger(
    __name__
)  # the __name__ resolve to "main" since we are at the root of the project.
# This will get the root logger since no logger in the configuration has this name.

coloredlogs.install(logger=logger)
logger.propagate = False


class DataFetcher:
    MAX_CONCURRENT_FETCHES = 10

    def __init__(
        self,
        req: GenerationAndCommitRequest | QuestionCreationRequest,
        openai_key,
        redis,
        task_id,
        questions=False,
    ):
        """
        Initial state is either through Redis (if older progress is available) or create a new task.

        Given that the number of valid results from LLM can be lesser than what is expected, it tries
        to schedule more batches once the first batch is done. The algorithm is kept simple over aggressively optimizing it.

        Since this happen only at the end of a single batch, it's a little slower than expected.
        The expected time for the entire task is less than 2 mins for 50 batches. (1000 samples)

        TODO: Data Store is currently redis. Which will become huge if not checked. Move that to a more stable DB.
        TODO: The next batch can be more aggressively scheduled. (Although that would be premature optimization)

        Parameters:
            req (GenerationAndCommitRequest): The request object for generation and commit.
            openai_key (str): The OpenAI API key.
            redis (str): The Redis connection.
            task_id (int): The task ID.

        Returns:
            None
        """
        self.req = req
        self.openai_key = openai_key
        self.redis = redis
        self.task_id = task_id
        self.data = {"data": []}
        self.semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_FETCHES)
        self.status = None
        self.iteration = 0
        self.questions = questions

    async def _initialize_from_redis(self):
        existing_data = await self.redis.hgetall(self.task_id)
        if existing_data and "data" in existing_data:
            self.data = json.loads(existing_data.get("data", self.data))
            logger.info("Found existing data for task %s", self.task_id)
            logger.info("Existing data total samples: %d", len(self.data))
        else:
            logger.info("No existing data found for task %s", self.task_id)
        self.status = existing_data

    async def _fetch_and_update(self, batch_index):
        async with self.semaphore:  # Acquire the semaphore
            try:
                if self.questions:
                    res = await utils.get_question(
                        self.openai_key,
                        self.req.num_samples if self.req.num_samples < 20 else 20,
                        self.req.content,
                        self.req.model,
                        self.req.system_prompt,
                        self.req.user_prompt,
                    )
                else:
                    res = await utils.get_data(
                        self.openai_key,
                        self.req.labels,
                        self.req.num_samples if self.req.num_samples < 20 else 20,
                        self.req.valid_data,
                        self.req.invalid_data,
                    )
                self.data["data"].extend(res)

                progress = min(100, len(self.data["data"]) / self.req.num_samples * 100)

                self.status = {
                    "status": "Processing",
                    "Progress": "%s%%" % progress,
                    "Detail": "Generating data (Batch %s)" % batch_index,
                    "data": json.dumps(self.data),
                }

                if self.questions:
                    self.status["content_row"] = self.req.index

                await self.redis.hset(
                    self.task_id,
                    mapping=self.status,
                )

                logger.info(
                    "Saved data to redis for task %s, iteration %d and Batch %s",
                    self.task_id,
                    self.iteration,
                    batch_index,
                )
            except Exception as e:
                logger.error(
                    "Failed to fetch data for task %s, iteration %d and Batch %s: %s",
                    self.task_id,
                    self.iteration,
                    batch_index,
                    str(e),
                )

    async def _fetch_data(self):
        await self._initialize_from_redis()
        tasks = []
        batch_size = 20
        num_samples = self.req.num_samples
        num_batches = max(
            1, (num_samples - len(self.data["data"]) + batch_size - 1) // batch_size
        )

        for batch_index in range(num_batches):
            task = self._fetch_and_update(batch_index)
            tasks.append(task)

        await asyncio.gather(*tasks)

        logger.info(
            "Iteration Completed. Current Total samples %d", len(self.data["data"])
        )

        if len(self.data["data"]) < num_samples:
            logger.info(
                "Need to fetch more data. Starting new iteration for task %s",
                self.task_id,
            )
            self.iteration += 1
            await self._fetch_data()

    async def fetch(self):
        await self._fetch_data()
        logger.info("Total samples downloaded %d", len(self.data["data"]))
        logger.info("All Data Fetched - Returning data")
        return self.data["data"]
