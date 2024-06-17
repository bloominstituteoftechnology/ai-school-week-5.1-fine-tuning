from langsmith.schemas import Run, Example
from langsmith.evaluation import evaluate
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langsmith.wrappers import wrap_openai
from openai import Client as OpenAIClient

# Define dataset: these are your test cases
dataset_name = "Fine-Tuning Dataset Example"

openai_client = wrap_openai(OpenAIClient())

# Define predictor functions
def predict_with_gpt_3_5_turbo(inputs: dict) -> dict:
    messages = [{"role": "user", "content": inputs['input'][1]['data']['content']}]
    response = openai_client.chat.completions.create(messages=messages, model="gpt-3.5-turbo")
    return {"output": response}

def predict_with_fine_tuned_model(inputs: dict) -> dict:
    messages = [{"role": "user", "content": inputs['input'][1]['data']['content']}]
    response = openai_client.chat.completions.create(messages=messages, model="") # enter tuned model here
    return {"output": response}

# Define evaluators
def check_for_valid_unit_test(run: Run, example: Example) -> dict:
    llm = ChatOpenAI()
    prompt = PromptTemplate(
        template=f"Determine if the following response is a valid unit test for the given function. Score 1 if it is valid and 0 if it is not. Only return the score.\n\nFunction:\n{example.inputs['function_code']}\n\nUnit Test:\n{example.outputs['completion']}",
        input_variables=["function_code", "completion"]
    )
    chain = prompt | llm | StrOutputParser()
    run_output = run.outputs['output'].choices[0].message.content
    response = chain.invoke({
        "function_code": example.inputs['function_code'],
        "completion": run_output
    })
    return {"key": "check_for_valid_unit_test", "score": 1 if "1" in response else 0}

baseline_experiment_results = evaluate(
    predict_with_gpt_3_5_turbo, # predictor
    data=dataset_name, # The data to predict and grade over
    evaluators=[check_for_valid_unit_test], # The evaluators to score the results
    experiment_prefix="unit-testing", # A prefix for your experiment names to easily identify them
    metadata={
      "version": "1.0.0",
    },
)

fine_tuned_experiment_results = evaluate(
    predict_with_fine_tuned_model, # predictor
    data=dataset_name, # The data to predict and grade over
    evaluators=[check_for_valid_unit_test], # The evaluators to score the results
    experiment_prefix="unit-testing", # A prefix for your experiment names to easily identify them
    metadata={
      "version": "1.0.0",
    },
)