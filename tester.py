from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.tools.shell.tool import ShellTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.tools import tool
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain_openai import ChatOpenAI
from prompt import *

@tool
def look_at_existing_app():
    """
    Retrieve all file names within the 'app' directory.

    This function loads the contents of the 'app' directory and extracts the file names.

    Returns:
        list: A list of file names present in the 'app' directory.
    """
    code = DirectoryLoader(f"./app/", silent_errors=True).load()
    return [c.metadata["source"] for c in code]

@tool
def get_page_contents(files):
    """
    Retrieve the current contents for the given file names.

    This function loads the contents of the specified files and formats them.
    The contents should be edited and not completely replaced.

    Args:
        files (list): A list of file names whose contents need to be retrieved.

    Returns:
        list: A list of formatted strings containing the file names and their contents.
    """
    loader = TextLoader(files[0])
    return [f"___{doc.metadata['source']}___\n{doc.page_content}" for doc in loader.load()]

@tool
def generate_unit_tests(function_code):
    """
    Generates the unit tests using OpenAI and `unittest` python library.
    """
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0,
    )

    system_prompt = SYSTEM_PROMPT

    prompt_str = f"""
    {system_prompt}
    
    ## Function to be tested:
    ```python
    {function_code}
    ```

    ## Generate comprehensive unit tests for the function above:
    """

    response = llm.invoke(prompt_str)
    return response

@tool
def update_file(file_path, new_content):
    """
    Add content to an existing file.
    """
    try:
        # Open the file in write mode
        with open(file_path, 'w') as file:
            # Write the new content to the file
            file.write(new_content)
        print("File updated successfully.")
    except Exception as e:
        print("Error:", e)

@tool
def create_new_file(filename: str):
    """Only use this if the file is not created already. 
    If it is, then use the tool get_page_contents. 
    Creates a new file with the given filename. 
    Only use this function when the file doesn't already exist."""
    with open(filename, 'w'):
        pass

# List of tools to use
tools = [
    ShellTool(ask_human_input=True),
    look_at_existing_app,
    get_page_contents,
    generate_unit_tests,
    create_new_file,
    update_file
    # Add more tools if needed
]


# Configure the language model
llm = ChatOpenAI(model="gpt-4o", temperature=0)


# Set up the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert web developer.",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


# Bind the tools to the language model
llm_with_tools = llm.bind_tools(tools)


agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)


agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# Main loop to prompt the user
while True:
    user_prompt = input("Prompt: ")
    list(agent_executor.stream({"input": user_prompt}))