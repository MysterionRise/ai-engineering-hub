from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

with open("data/generated_data.jsonl", "rb") as f:
    client.files.create(file=f, purpose="fine-tune")

print(client.files.list())

client.fine_tuning.create(training_file="file-xYgYu76YizAh1t9hrAxTPeYc", model="gpt-3.5-turbo")
#
# # List 10 fine-tuning jobs
print(client.fine_tuning.list(limit=10))
