# llm_qa_generation.py
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

api_key = 'YOUR_KEY'

from typing import List

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

model = ChatOpenAI(openai_api_key=api_key, temperature=0, model_name="gpt-3.5-turbo")


# Define your desired data structure.
# Here's another example, but with a compound typed field.
class Actor(BaseModel):
    name: str = Field(description="name of an actor")
    film_names: List[str] = Field(description="list of names of films they starred in")


actor_query = "Generate the filmography for a random actor."

parser = PydanticOutputParser(pydantic_object=Actor)
prompt_msg = parser.get_format_instructions()
prompt = ChatPromptTemplate.from_template("{query}")
parser = StrOutputParser()
chain = prompt | model | parser

response = chain.invoke({"query": actor_query+"\n"+prompt_msg})
print(response)