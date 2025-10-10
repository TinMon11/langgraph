import os

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI


def main():
    print("Hello, World!")
    print("OS ENVIERON", os.getenv("OPENAI_API_KEY"))


if __name__ == "__main__":
    main()
