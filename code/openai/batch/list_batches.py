from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# List recent batches
batches = client.batches.list(limit=10)
print("\nRecent batches:")
for batch in batches:
    print(f"\nBatch ID: {batch.id}")
    print(f"Status: {batch.status}")
    print(f"Created at: {batch.created_at}")
    print("-" * 50) 