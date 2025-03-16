def print_welcome_message():
    """Display welcome message and instructions."""
    print("\nWelcome to the Document QA System!")
    print("Type your questions and I'll answer based on the available documents.")
    print("Type 'quit' or 'exit' to end the session.\n")

def is_exit_command(user_input):
    """Check if user wants to exit the chat."""
    return user_input.lower().strip() in ['quit', 'exit']

def format_response(response, sources):
    """Format the final response with sources."""
    return f"\nAnswer: {response}\n{sources}\n"
