from dotenv import load_dotenv
load_dotenv()


from importlib.metadata import version

lg_version = version("langgraph")
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


print("langchain core version ", version("langchain_core"))
print("langgraph core version ", lg_version)


def main():
    print("Hello, World!")
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

  
if __name__ == "__main__":
    main()