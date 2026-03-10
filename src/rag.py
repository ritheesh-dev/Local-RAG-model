import sys
from pdf_to_vector import pdf_to_vector
from question_vector import ask_question

def main():
    print("--- Local PDF RAG System ---")
    print("1. Process new PDF (Create Vector DB)")
    print("2. Ask a question (Query existing DB)")
    
    choice = input("Select an option (1/2): ")

    if choice == '1':
        pdf_file = input("Enter the PDF filename (e.g., project.pdf): ")
        pdf_to_vector(pdf_file)
    elif choice == '2':
        query = input("Ask your question: ")
        answer = ask_question(query)
        print("\n--- AI RESPONSE ---")
        print(answer)
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()