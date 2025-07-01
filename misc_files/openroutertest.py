from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from os import getenv
from dotenv import load_dotenv

load_dotenv()

template = """Question: {question}
Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm = ChatOpenAI(
  openai_api_key=getenv("OPENROUTER_API_KEY"),
  openai_api_base=getenv("OPENROUTER_BASE_URL"),
  model_name="deepseek/deepseek-chat-v3-0324:free",
)

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

chain = prompt | llm
print(chain.invoke({"question": question}))
