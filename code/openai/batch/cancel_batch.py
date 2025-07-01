from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Cancel the batch
batch_id = "batch_68482de18e748190a84c1cf0f890bc0b"
batch = client.batches.retrieve(batch_id)
batch.cancel()
print(f"Batch status: {batch.status}") 