from openai import OpenAI
from pydantic import BaseModel
from enum import Enum

class Receipt(BaseModel):
    receipt_number: str
    destination_country: str
    transportation_mode: str
    total_weight: str
    number_of_boxes: str
    export_date: str
    is_blur: bool = None


json_schema = Receipt.model_json_schema()

def process_document_with_ai(document_text):
    # Initialize the client pointing to your local server
    client = OpenAI(
        base_url="",
        api_key=""  #
    )
    
    # Make the API call
    response = client.chat.completions.create(
        model="Qwen3",
        messages=[
        {"role": "system", "content": """You are a helpful assistant. Extract information EXACTLY as it appears in the provided text, without combining with other texts. 
        Return only a single valid JSON object with the shipping details, without any additional text, comments, or trailing content"""},
        {"role": "user", "content": f"""
        Extract ONLY the following fields from the text below and return in json format:
        - P.Q.7 receipt number
        - Destination country
        - Transportation mode
        - Total weight
        - Number of boxes
        - Export date

        CONTEXT:
        1. Process ONLY the text provided in this single request
        2. Receipt number should have format NP****
        3. For destination country, preserve the EXACT text format - do not separate or modify country names
        For example, if text contains 'Youyiguan CHINAQuang Binh VIETNAM', return 'Youyiguan CHINA, Quang Binh VIETNAM'
        Do not consider phrases like 'IMPORT AND EXPORT TRADE' as destination countries
        4. Transportation may be start with by, for example: By Train
        5. Number of boxes maybe have unit of CARTONS (cartons)
        6. Export date should be near to phrase: Date of exportation and have format dd/mm/yyyy

        Return only a single valid JSON object with the shipping details, without any additional text, comments, or trailing content

        Text to process:
        {document_text}
        """}
    ],
    temperature=0.2,
    max_tokens=2048,
    extra_body={"guided_json": json_schema},
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