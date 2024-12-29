import os
import sys
import signal
from typing import Optional, Literal
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.schema.language_model import BaseLanguageModel
from ingestion import validate_environment
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama

# Update the ModelChoice type
ModelChoice = Literal["openai", "claude", "gemini", "ollama"]

def signal_handler(_, __):
    """Handle graceful exit on CTRL+C."""
    print("\nGoodbye!")
    sys.exit(0)

def select_model() -> ModelChoice:
    """Prompt user to select the chat model."""
    while True:
        print("\nAvailable models:")
        print("1. OpenAI GPT")
        print("2. Anthropic Claude")
        print("3. Google Gemini")
        print("4. Ollama (Local)") # Add new option
        choice = input("\nSelect a model (1, 2, 3, or 4): ").strip()

        if choice == "1":
            return "openai"
        elif choice == "2":
            return "claude"
        elif choice == "3":
            return "gemini"
        elif choice == "4":
            return "ollama"
        else:
            print("Invalid choice. Please select 1, 2, 3, or 4.")

def get_language_model(model_choice: ModelChoice) -> BaseLanguageModel:
    """Initialize the selected language model."""
    if model_choice == "openai":
        return ChatOpenAI()
    elif model_choice == "claude":
        return ChatAnthropic(
            model_name="claude-3-sonnet-20240229",
            temperature=0.7,
            timeout=120,
            stop=["Human:", "Assistant:"]
        )
    elif model_choice == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.7,
            convert_system_message_to_human=True
        )
    else:
        return ChatOllama(
            model="llama3",
            temperature=0.7
        )

def setup_chain(model_choice: ModelChoice):
    """Initialize and configure the retrieval chain."""
    embeddings = OpenAIEmbeddings()
    llm = get_language_model(model_choice)

    vectorstore = PineconeVectorStore(
        index_name=os.environ["PINECONE_INDEX_NAME"],
        embedding=embeddings
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    return create_retrieval_chain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=combine_docs_chain
    )

def get_response(chain, question: str) -> Optional[str]:
    """Get response from the chain for a given question."""
    try:
        result = chain.invoke({"input": question})
        return result["answer"]
    except (ValueError, KeyError) as e:
        print(f"\nError processing response: {str(e)}")
        return None
    except ConnectionError as e:
        print(f"\nConnection error: {str(e)}")
        return None

def main():
    """Main chat loop."""
    load_dotenv()
    validate_environment()

    print("Welcome to the RAG Chat System!")
    model_choice = select_model()
    print(f"\nInitializing chat system with {model_choice.upper()}...")

    chain = setup_chain(model_choice)
    print("Chat initialized! Type 'quit' or 'exit' to end the conversation.")

    while True:
        try:
            user_input = input("\nYour question: ").strip()

            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break

            if not user_input:
                continue

            print("Retrieving answer...")
            answer = get_response(chain, user_input)
            if answer:
                print("\nAnswer:", answer)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except (ConnectionError, ValueError) as e:
            print(f"\nError occurred: {str(e)}")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    main()
