from openai import OpenAI

def process_document_with_ai(document_text):
    # Initialize the client pointing to your local server
    client = OpenAI(
        base_url="",
        api_key=""  # Your local server may not require an API key
    )
    
    # Make the API call
    response = client.chat.completions.create(
        model="microsoft/Phi-4-mini-instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Extract the following information (P.Q.7 receipt number, Destination country, Transportation mode, Total weight, Number of boxes, Export date) from this document text:\n\n{document_text}"}
        ]
    )
    
    # Return the AI response
    return response.choices[0].message.content

# Alternatively, use the Python function directly for more controlled extraction
def main():
    document_text = """Your full document text here"""
    
    # Extract using our custom function
    extracted_info = extract_document_info(document_text)
    print("Extracted with Python function:", extracted_info)
    
    # Or use the AI model
    ai_extraction = process_document_with_ai(document_text)
    print("Extracted with AI model:", ai_extraction)

if __name__ == "__main__":
    main()