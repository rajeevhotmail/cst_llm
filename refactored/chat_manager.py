import json
from datetime import datetime
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage


def get_chat_response(query, relevant_docs, chat_history=None):
    """Get response from ChatGPT using query and relevant documents."""
    chat = ChatOpenAI(temperature=0)

    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    messages = [
        SystemMessage(content="You are a helpful assistant. Use the provided context to answer questions."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
    ]

    if chat_history:
        messages.extend(chat_history)

    response = chat(messages)
    return response.content

def save_conversation(conversation, filename=None):
    """Save chat conversation to a JSON file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_history_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump(conversation, f, indent=2)
    return filename

def load_conversation(filename):
    """Load chat conversation from a JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def format_sources(docs):
    """Format source documents for citation."""
    sources = []
    for doc in docs:
        if hasattr(doc, 'metadata') and 'source' in doc.metadata:
            sources.append(doc.metadata['source'])
    return "\nSources:\n" + "\n".join(sources) if sources else ""

def get_mock_response(query, relevant_docs):
    """Display context with cleaner formatting"""
    context = "\n=== RELEVANT CONTEXT ===\n"
    for i, doc in enumerate(relevant_docs, 1):
        # Get just the filename from the full path
        filename = doc.metadata.get('source', '').split('/')[-1]

        # Clean up the content by removing extra newlines
        content = doc.page_content.strip().replace('\n\n\n', '\n')

        context += f"\n[{i}] File: {filename}\n"
        context += f"{'=' * 50}\n"
        context += f"{content}\n"

    context += "\n=== END CONTEXT ===\n"
    return context