import json
import logging

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
    labels,
    api_key,
    num_samples,
    model="gpt-3.5-turbo",
    temperature=1,
    max_tokens=None,
    valid_data=None,
    invalid_data=None,
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

    user_content = f"Generate {num_samples} robust samples"
    template = ""

    if valid_data:
        valid_data = json.dumps(valid_data, separators=(",", ":")).replace("},", "},\n")
        template = template + "The correctly labeled data is \n {valid_data}. \n "
    if invalid_data:
        invalid_data = json.dumps(invalid_data, separators=(",", ":")).replace(
            "},", "},\n"
        )
        template = template + "The incorrectly labeled data is \n {invalid_data}.\n "
    template = (
        template
        + "{format_instructions} \n {text}. \n The valid labels are {labels}. \n "
    )
    template = (
        template
        + "Provide an output that can be directly parsed by json.loads and provide JSON only. NO CONTEXT."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "You are a helpful data generation assistant. You generate labeled data for other models to learn from."
                    "You only generate data for 'valid labels' that you are given. You are given a question and a list of valid labels."
                    "You maybe also be given a list of correctly labeled data and incorrectly labeled data. You use these to improve the data generation."
                    "The data generated should be in the format of a list of dictionaries, where each dictionary has the keys 'text' and 'label'."
                    "You enclose the JSON values with double quotes"
                )
            ),
            HumanMessagePromptTemplate.from_template(template),
        ]
    )

    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

    output = await chain.arun(
        text=user_content,
        labels=labels,
        invalid_data=invalid_data,
        valid_data=valid_data,
        format_instructions=parser.get_format_instructions(),
    )

    # output = llm(_input.to_messages())
    try:
        parsed = parse(output, parser)
        if len(parsed) < num_samples:
            logger.error("Only got %d responses", len(parsed))
    except Exception as e:
        logger.error("Exception in parsing %s", str(e))
        logger.error("Corrupted JSON: " + output)
        parsed = []

    return parsed


async def get_data(
    api_key,
    labels,
    num_samples,
    valid_data=None,
    invalid_data=None,
):
    try:
        res = await generate(
            labels,
            api_key,
            num_samples,
            model="gpt-3.5-turbo",
            temperature=1,
            max_tokens=None,
            valid_data=valid_data,
            invalid_data=invalid_data,
        )
        return res
    except Exception as e:
        logger.error("Error %e", e)
    return []


def split_data(res, split):
    assert sum(split) == 100, "Split should sum to 100"
    train_end = int(len(res) * split[0] / 100)
    val_end = train_end + int(len(res) * split[1] / 100)
    test_end = val_end + int(len(res) * split[2] / 100)
    train = res[:train_end]
    val = res[train_end:val_end]
    test = res[val_end:test_end]
    return train, val, test
