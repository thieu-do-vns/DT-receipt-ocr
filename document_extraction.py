import os
import tempfile
import requests
from typing import Dict, Any, Union, List, Tuple
from urllib.parse import urlparse
import json
import re

# Import the existing extraction functions
from extract_text import extract_all_english_text, filter_specific_fields, extract_regions_from_image, extract_text_from_region

def extract_total_weight(bboxes):
    potential = []
    anchor_box = []
    res = []
    unit = ''
    for i, bbox in enumerate(bboxes):
        if bbox['text'].lower() == 'quantity':
            potential = bboxes[i:]
            anchor_box = bboxes[i]
        if '.0000' in bbox['text']:
            unit = bbox['text']

    for bbox in potential:
        if is_overlap(bbox['bbox'], anchor_box['bbox']) and bbox['text'] != anchor_box['text'] and is_not_second_row(bbox):
            return ''.join(char for char in bbox['text'] if char.isdigit()) + ',' + unit
    return ''

def is_overlap(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2

    if (x1 <= x3 <= x2 or x1 <= x4 <= x2) and (y2 < y3 or y4 < y1):
        return True
    return False

def is_not_second_row(bbox):
    return '.0000' not in bbox['text']

def flatten_dict_list(data):
    return [item for k,v in data.items() for item in v]

def extract_epxorted_date(bboxes):
    # Pattern để trích xuất định dạng dd/mm/yyyy
    pattern = r'\d{2}/\d{2}/\d{4}'
    
    # Tìm tất cả các kết quả khớp
    for box in bboxes:
        dates_found = re.findall(pattern, box['text'])
        if dates_found:
            print(f"Tìm thấy {len(dates_found)} ngày tháng:")
            for date in dates_found:
                print(date)
            return dates_found
    return None

def download_image_from_url(image_url: str) -> str:
    """
    Download an image from URL to a temporary file
    
    Args:
        image_url: URL of the image
        
    Returns:
        str: Path to the downloaded image
    """
    # Validate URL
    parsed_url = urlparse(image_url)
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(f"Invalid URL: {image_url}")
    
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_path = temp_file.name
    temp_file.close()
    
    try:
        # Download the image
        response = requests.get(image_url, stream=True, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return temp_path
    except Exception as e:
        # Clean up in case of error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise Exception(f"Failed to download image: {str(e)}")


def save_base64_image(base64_image: str) -> str:
    """
    Save a base64-encoded image to a temporary file
    
    Args:
        base64_image: Base64-encoded image string
        
    Returns:
        str: Path to the saved image file
    """
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_path = temp_file.name
    temp_file.close()
    
    try:
        # Check if the string contains the data URI prefix and remove it if present
        if ',' in base64_image:
            # Format is typically: "data:image/jpeg;base64,/9j/4AAQSkZ..."
            base64_image = base64_image.split(',', 1)[1]
        
        # Decode the base64 string
        image_data = base64.b64decode(base64_image)
        
        # Save to the temporary file
        with open(temp_path, 'wb') as f:
            f.write(image_data)
        
        return temp_path
    except Exception as e:
        # Clean up in case of error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise Exception(f"Failed to decode and save base64 image: {str(e)}")

def extract_fields_by_region_wrapper(image_path: str) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    """
    Extract document fields by region - modified version of the original function to return both fields and region texts
    
    Args:
        image_path: Path to the image
        
    Returns:
        Tuple containing fields dictionary and region texts dictionary
    """
    # Extract regions from the image
    regions = extract_regions_from_image(image_path)

    # Extract text from each region
    region_texts = {}
    for region_name, region_image in regions.items():
        region_texts[region_name] = extract_text_from_region(region_image, region_name)
    
    # # Combine all text from regions for field extraction
    # all_text = []
    # for region_name, texts in region_texts.items():
    #     all_text.extend(texts)
    
    # # Extract fields from all text
    # fields = filter_specific_fields(all_text)
    
    # return fields, region_texts
    return region_texts

def extract_document(image: str, use_regions: bool = True) -> Dict[str, Any]:
    """
    Extract document information from an image (URL or base64)
    Args:
        image: URL of the image or base64-encoded image string
        use_regions: Whether to use region-based extraction for better results
    Returns:
        Dictionary containing extracted fields and raw text
    """
    try:
        # # Check if input is URL or base64
        # parsed_url = urlparse(image)
        # if parsed_url.scheme and parsed_url.netloc:
        #     # It's a URL
        #     local_path = download_image_from_url(image)
        # else:
        #     # Assume it's a base64 string
        #     local_path = save_base64_image(image)

        local_path = image
            
        result = {
            'status': 'success',
            'fields': {},
            'region_texts': {},
            'raw_text': []
        }
        
        try:
            if use_regions:
                # Extract text by regions
                region_texts = extract_fields_by_region_wrapper(local_path)
                result['region_texts'] = region_texts
                # Combine all text for raw_text
                all_text = []
                for texts in region_texts.values():
                    all_text.extend(texts)
                result['raw_text'] = all_text
            else:
                # Extract all text at once
                all_text = extract_all_english_text(local_path)
                # Add raw text to results
                result['raw_text'] = all_text
                # Filter for specific fields
                # result['fields'] = filter_specific_fields(all_text)
            
            return result
        finally:
            # Clean up the downloaded image
            if os.path.exists(local_path):
                # os.remove(local_path)
                pass
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'fields': {},
            'raw_text': []
        }

def extract_with_llm(document_text: str) -> Dict[str, Any]:
    """
    Use LLM to extract information from document text
    
    Args:
        document_text: Text extracted from the document
        
    Returns:
        Dictionary containing extracted fields
    """
    from llm_extract import process_document_with_ai
    
    try:
        # Convert document_text to a string if it's a list
        if isinstance(document_text, list):
            text_str = "\n".join([item.get('text', '') for item in document_text if isinstance(item, dict) and 'text' in item])
        else:
            text_str = document_text
        
        # Process text with AI
        ai_extraction = process_document_with_ai(text_str)
        
        return {
            'status': 'success',
            'llm_extraction': ai_extraction
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'llm_extraction': None
        }

def extract_info(image: str):
    """
    Main function to extract information from an image URL
    
    Args:
        image_url: URL of the image
        
    Returns:
        Extracted information from the document
    """
    # Extract document information
    ocr_result = extract_document(image)
    
    if ocr_result['status'] == 'success':
        ocr_text = "=== EXTRACTED FIELDS ===\n"  # Fixed string quote
        for field_name, field_value in ocr_result['region_texts'].items(): 
            ocr_text += f"{field_name}: {field_value}\n"  # Changed print() to string concatenation, added newline
        
        total_weight = extract_total_weight(flatten_dict_list(ocr_result['region_texts']))
        export_date = extract_epxorted_date(ocr_result['region_texts']['middle'])

        print("[ocr text]: ", ocr_text)

        llm_result = extract_with_llm(ocr_text) 
        print(llm_result)
        if llm_result['status'] == 'success':
            try:
                llm_result['llm_extraction'] = json.loads(llm_result['llm_extraction'])
                llm_result['llm_extraction']['total_weight_heuristic'] = total_weight
                return llm_result['llm_extraction']  # Return the LLM extraction
            except:
                return "Error: Could not parse llm response"
        else:
            return f"Error: {llm_result['error']}"
    else:
        return f"Error: {ocr_result['error']}"  # Changed 'result' to 'ocr_result'

# Main execution
if __name__ == "__main__":
    # Replace with your image path
    # image_path = "ocr_test_4_page-0001.jpg"
    # image_path = 'rotate.jpg'
    # image_path = 'processed_certificate.jpg'
    image_path = 'test_image/d2.JPG'
    
    # Extract fields by region
    region_texts = extract_info(image_path)
    print(region_texts)
