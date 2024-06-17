from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from functools import partial
import concurrent.futures
import json

llm = ChatOpenAI(model="gpt-3.5-turbo")
# llm = ChatOpenAI(model="ft:gpt-3.5-turbo-0613:bloomtech::9bFk26Dq") # already tuned model

def generate_unit_test(function_code):
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessage(content="You are an AI that generates unit tests for Python functions."),
            HumanMessage(content=f"Generate a unit test for this function: \n```python\n{function_code}\n```")
        ]
    )
    chain = prompt | llm
    response = chain.invoke({})
    return {
        "prompt": f"Generate a unit test for this function: \n```python\n{function_code}\n```",
        "completion": response.content  # Access the content attribute of AIMessage
    }

functions_to_test = [
    "def subtract(a, b):\n    return a - b",
    "def multiply(a, b):\n    return a * b",
    "def divide(a, b):\n    if b == 0:\n        raise ValueError('Cannot divide by zero')\n    return a / b",
    "def is_even(n):\n    return n % 2 == 0",
    "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n - 1)"
]

generate_synthetic_data = partial(generate_unit_test)

with concurrent.futures.ThreadPoolExecutor() as executor:
    synthetic_data = list(executor.map(generate_synthetic_data, functions_to_test))

# Output the synthetic data
print(json.dumps(synthetic_data, indent=4))
