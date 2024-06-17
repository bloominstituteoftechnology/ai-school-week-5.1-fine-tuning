from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from functools import partial
import concurrent.futures

llm = ChatOpenAI(model="gpt-4o")

def generate_unit_test(prompt_tag, function_code):
    prompt = ChatPromptTemplate(
        tags=["unit_tests", prompt_tag],
        messages=[
            SystemMessage(content="You are an AI that generates unit tests for Python functions."),
            HumanMessage(content=f"Generate a unit test for this function: \n```python\n{function_code}\n```")
        ]
    )
    chain = prompt | llm
    response = chain.invoke({})
    return {
        "prompt": prompt.messages[1].content,
        "completion": response['choices'][0]['message']['content'].strip()
    }

functions_to_test = [
    "def subtract(a, b):\n    return a - b",
    "def multiply(a, b):\n    return a * b",
    "def divide(a, b):\n    if b == 0:\n        raise ValueError('Cannot divide by zero')\n    return a / b",
    "def is_even(n):\n    return n % 2 == 0",
    "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n - 1)"
]

generate_synthetic_data = partial(generate_unit_test, prompt_tag="unit_test")

with concurrent.futures.ThreadPoolExecutor() as executor:
    synthetic_data = list(executor.map(generate_synthetic_data, functions_to_test))

# Output the synthetic data
import json
print(json.dumps(synthetic_data, indent=4))
