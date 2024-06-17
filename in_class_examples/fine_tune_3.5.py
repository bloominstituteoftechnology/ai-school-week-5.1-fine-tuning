# Import necessary modules
import os
import sys
import json
from openai import OpenAI

# Initialize the OpenAI client with the API key from environment variables
client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],
)

# Get the root directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the root directory to the system path
sys.path.append(root_dir)

# Now you can import the variables from data.py
from data import *

# Define the names of the training and validation data files
training_file_name = "training_data.jsonl"
validation_file_name = "validation_data.jsonl"

# Function to prepare data and write it to a JSONL file
def prepare_data(dictionary_data, final_file_name):
    with open(final_file_name, 'w') as outfile:
        for entry in dictionary_data:
            json.dump(entry, outfile)
            outfile.write('\n')

# Call the prepare_data function for both training and validation data
prepare_data(training_data, training_file_name)
prepare_data(validation_data, validation_file_name)

# Upload the training data file to OpenAI and get the file ID
training_file_id = client.files.create(
  file=open(training_file_name, "rb"),
  purpose="fine-tune"
)

# Upload the validation data file to OpenAI and get the file ID
validation_file_id = client.files.create(
  file=open(validation_file_name, "rb"),
  purpose="fine-tune"
)

# Print the file IDs for reference
print(f"Training File ID: {training_file_id}")
print(f"Validation File ID: {validation_file_id}")

# Create a fine-tuning job with the uploaded files and specific hyperparameters
response = client.fine_tuning.jobs.create(
  training_file=training_file_id.id, 
  validation_file=validation_file_id.id,
  model="gpt-3.5-turbo-0613", 
  hyperparameters={
    "n_epochs": 15,
    "batch_size": 3,
    "learning_rate_multiplier": 0.3
  }
)

# Retrieve the job ID and status from the response
job_id = response.id
status = response.status

# Print the job ID and initial status
print(f'Fine-tuning model with jobID: {job_id}.')
print(f"Training Response: {response}")
print(f"Training Status: {status}")

# Import signal and datetime modules for handling interruptions and timestamps
import signal
import datetime

# Define a signal handler to manage interruptions
def signal_handler(sig, frame):
    status = client.fine_tuning.jobs.retrieve(job_id).status
    print(f"Stream interrupted. Job is still {status}.")
    return

# Print the start of event streaming
print(f"Streaming events for the fine-tuning job: {job_id}")

# Set up the signal handler for keyboard interruptions
signal.signal(signal.SIGINT, signal_handler)

# List events for the fine-tuning job and print them with timestamps
events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id)
try:
    for event in events:
        print(
            f'{datetime.datetime.fromtimestamp(event.created_at)} {event.message}'
        )
except Exception:
    print("Stream interrupted (client disconnected).")

# Import time module for sleep function
import time

# Check the status of the fine-tuning job and wait if it is not in a terminal state
status = client.fine_tuning.jobs.retrieve(job_id).status
if status not in ["succeeded", "failed"]:
    print(f"Job not in terminal status: {status}. Waiting.")
    while status not in ["succeeded", "failed"]:
        time.sleep(2)
        status = client.fine_tuning.jobs.retrieve(job_id).status
        print(f"Status: {status}")
else:
    print(f"Finetune job {job_id} finished with status: {status}")

# Print the status of other fine-tuning jobs in the subscription
print("Checking other fine-tune jobs in the subscription.")
result = client.fine_tuning.jobs.list()
print(f"Found {len(result.data)} fine-tune jobs.")

# Retrieve and print the ID of the fine-tuned model
fine_tuned_model = result.data[0].fine_tuned_model
print(fine_tuned_model)

