import os
import json
import openai
import dotenv


def load_openai_key(path):
    dotenv.load_dotenv(path)
    openai.api_key = os.getenv("OPENAI_API_KEY")


def build_messages_from_file(path, prompt):
    messages = json.load(open(path, "r"))
    messages.append({"role": "user", "content": prompt})
    return messages
