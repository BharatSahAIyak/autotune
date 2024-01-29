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


def parse(output, parser, question=False):
    """
    Parses the given output using dirtyjson and returns a list of dictionaries.

    Args:
    output (str): The output to be parsed.

    Returns:
    list: A list of dictionaries containing the parsed data.
    """
    parsed = dirtyjson.loads(output)
    if question:
        parsed = parsed["qna"]
    else:
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


async def get_question(api_key, num_samples, content, multiple_chunks):
    try:
        res = await generate_questions(
            api_key,
            num_samples,
            content,
            multiple_chunks,
            model="gpt-4-1106-preview",
            temperature=1,
            max_tokens=None,
        )
        return res
    except Exception as e:
        logger.error("Error %e", e)
    return []


class QuestionDataset(BaseModel):
    question: str = Field(..., description="The question generated")
    answer: str = Field(
        ...,
        description="The answer generated",
    )


async def generate_questions(
    api_key,
    num_samples,
    content,
    multiple_chunks,
    model="gpt-4-1106-preview",
    temperature=1,
    max_tokens=None,
):
    parser = PydanticOutputParser(pydantic_object=QuestionDataset)

    llm = ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    user_content = f"Generate {num_samples} robust samples"
    template = ""

    template = template + "Required Output: \n"
    template = (
        template + "- A list of Q&A pair in JSON format. Generate only 1 question \n"
    )
    if multiple_chunks:
        template = (
            template
            + "Your task is to analyize the following two chunks of text and generate questions and answers.\n"
        )
        template = (
            template
            + "Generate a SINGLE question covering BOTH chunks. The context of the question should be broad enough to COVER BOTH CHUNKS \n"
        )
        template = (
            template
            + "The question can be generalized a little so that there is coverage of information from BOTH CHUNKS \n"
        )
    else:
        template = (
            template
            + "Your task is to analyize the following text and generate questions and answers.\n"
        )
    template = template + f"{content} \n \n"
    template = (
        template
        + " The questions generated shoud be UNIQUE, CRYSTAL CLEAR, and INDEPENDANT of the text given. Use PRECISE TERMS in the questions \n"
    )
    template = (
        template
        + " Don't have terms which reference the text. Questions should be understood by a 3rd party with no context.\n"
    )
    template = (
        template
        + "Keep the answers to the questions detailed. Avoid one word answers. Add all relelvant information to the answer. \n"
    )
    template = (
        template
        + "Answer as a leading industry expert explaining to a 7year old with no understanding of the question. \n"
    )
    template = (
        template
        + "Provide an output that can be directly parsed by `json.loads` and provide JSON output only. NO CONTEXT."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "You are a helpful data generation assistant.You create questions that are answered by an answer that combines information from both chunks chunk1 and chunk2."
                    "You create a set of questions and answers based on the given text assuming they will be asked by a farmer. You are an expert in this field"
                    "Each question should closely match the language, terminology, and key phrases used in the text to ensure a high content overlap."
                    "Focus on extracting and reformulating specific sentences or phrases from the text as questions, and provide direct quotes or slight rephrasings from the text as answers."
                    "Ask questions that have longer answers. Keep the answers verbose."
                    "Aim for a content overlap of about 90%. Overlap is defined as the entire overlap with all the questions stating the coverage of the over Q and A with the text."
                    "You enclose the JSON values with double quotes ALWAYS."
                    "Don't say terms like 'as mentioned in the text'."
                    "The Questions should not contain vage terms like 'process' or 'programme' or 'product'."
                    "Don't reference images like 'Fig. 29.1'. Generate questions independent of such lines"
                    "Use complete terms in the questions and answers.  Don't assume the reader to have reference to anything other than the question"
                    "The Required Ouput JSON format is as follows: \n"
                    "{ \n"
                    '   "qna": [ \n'
                    "       { \n"
                    '           "question": "<QUESTION_TEXT>", \n'
                    '           "answer": "<ANSWER_TEXT>" \n'
                    "       }, \n"
                    "       ... \n"
                    "   ] \n"
                    "}"
                )
            ),
            HumanMessagePromptTemplate.from_template(template),
        ]
    )

    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

    output = await chain.arun(
        text=user_content,
        content=content,
        format_instructions=parser.get_format_instructions(),
    )

    # output = llm(_input.to_messages())

    try:
        parsed = parse(output, parser, True)
        if len(parsed) < num_samples:
            logger.error("Only got %d responses", len(parsed))
    except Exception as e:
        logger.error("Exception in parsing %s", str(e))
        logger.error("Corrupted JSON: " + output)
        parsed = []

    return parsed


def split_data(res, split):
    assert sum(split) == 100, "Split should sum to 100"
    train_end = int(len(res) * split[0] / 100)
    val_end = train_end + int(len(res) * split[1] / 100)
    test_end = val_end + int(len(res) * split[2] / 100)
    train = res[:train_end]
    val = res[train_end:val_end]
    test = res[val_end:test_end]
    return train, val, test
