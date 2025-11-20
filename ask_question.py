# ======================================================================================
# Interactive Question Input Script
#
# Description:
# This script provides a user-friendly way to ask a multi-line question. It reads
# input from the user until they signal the end (by pressing Ctrl+D on a new line)
# and saves the question to a temporary file.
#
# This avoids shell quoting issues when passing complex questions to 'make'.
# ======================================================================================

import sys
import os

def main():
    """
    Reads a multi-line question from stdin and saves it to a temporary file.
    """
    question_file = "logs/last_question.txt"
    
    print("Please enter your question. Press Ctrl+D on a new line when you are done.")
    print("-" * 20)
    
    # Read multi-line input from the user
    question_text = sys.stdin.read().strip()
    
    if not question_text:
        print("\nNo input received. Exiting.")
        return
        
    # Ensure the logs directory exists
    os.makedirs(os.path.dirname(question_file), exist_ok=True)
    
    # Save the question to the file
    with open(question_file, "w") as f:
        f.write(question_text)
        
    print(f"\nQuestion saved to {question_file}")
    print("You can now run 'make test-from-file' to run inference.")

if __name__ == "__main__":
    main()