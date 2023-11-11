import json
import logging
import time

import coloredlogs
import dirtyjson
import langchain
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema.messages import SystemMessage
from pydantic import BaseModel, Field

langchain.debug = False

# setup loggers
# logging.config.fileConfig("logging.conf", disable_existing_loggers=False)

# get root logger
logger = logging.getLogger(
    __name__
)  # the __name__ resolve to "main" since we are at the root of the project.
# This will get the root logger since no logger in the configuration has this name.

coloredlogs.install(logger=logger)
logger.propagate = False


USER_CONTENT = "Generate 20 robust samples."


def parse(output, parser):
    """
    Parses the given output using dirtyjson and returns a list of dictionaries.

    Args:
    output (str): The output to be parsed.

    Returns:
    list: A list of dictionaries containing the parsed data.
    """
    parsed = dirtyjson.loads(output)
    parsed = list(parsed)
    data = []
    for i in parsed:
        try:
            parser.parse(json.dumps(dict(i)))
            data.append(dict(i))
        except Exception as e:
            logger.error(e, dict(i))
    return data


async def generate(
    user_content,
    system_content,
    labels,
    api_key,
    model="gpt-3.5-turbo",
    temperature=1,
    max_tokens=None,
):
    class LabeledDataset(BaseModel):
        text: str = Field(..., description="The text of the sample generated")
        label: str = Field(
            ...,
            description="The label of the sample generated. Valid labels are {labels}".format(
                labels=labels
            ),
        )

    parser = PydanticOutputParser(pydantic_object=LabeledDataset)

    llm = ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "You are a helpful data generation assistant. You generate labeled data for other models to learn from."
                    "You only generate data for 'valid labels' that you are given. You are given a question and a list of valid labels."
                    "The data generated should be in the format of a list of dictionaries, where each dictionary has the keys 'text' and 'label'."
                    "You enclose the JSON values with double quotes"
                )
            ),
            HumanMessagePromptTemplate.from_template(
                "{format_instructions} \n {text}. \n The valid labels are [{labels}]."
            ),
        ]
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    output = await chain.arun(
        text=user_content,
        labels=labels,
        format_instructions=parser.get_format_instructions(),
    )

    # output = llm(_input.to_messages())
    try:
        parsed = parse(output, parser)
        if len(parsed) < 20:
            logger.error("Only got %d responses", len(parsed))
    except Exception as e:
        logger.error("Exception in parsing %s", str(e))
        parsed = []

    return parsed


async def get_data(system_content, api_key, task, labels, num_labels=None):
    time_stamp = str(int(time.time()))
    user_content = f"Timestamp = {time_stamp}, {USER_CONTENT}"
    try:
        res = await generate(user_content, system_content, labels, api_key)
        return res
    except Exception as e:
        logger.error("Error %e", e)
    return []


def validate_data(res, task, num_labels):
    for item in res:
        if task == "seq2seq":
            if "Input" not in item or "Output" not in item:
                return False
        elif task == "classification":
            if (
                "text" not in item
                or "label" not in item
                or int(item["label"]) >= num_labels
                or int(item["label"]) < 0
            ):
                return False
    return True


def split_data(res, split):
    assert sum(split) == 100, "Split should sum to 100"
    train_end = int(len(res) * split[0] / 100)
    val_end = train_end + int(len(res) * split[1] / 100)
    test_end = val_end + int(len(res) * split[2] / 100)
    train = res[:train_end]
    val = res[train_end:val_end]
    test = res[val_end:test_end]
    return train, val, test


def get_cols(task):
    if task == "seq2seq":
        return ["Input", "Output"]
    elif task == "text_classification":
        return ["text", "label"]
