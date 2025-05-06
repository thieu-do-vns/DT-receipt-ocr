import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
import re

def extract_all_english_text(image_path):
    """
    Extract all English text from the document using PaddleOCR
    
    Args:
        image_path (str): Path to the document image
        
    Returns:
        list: List of dictionaries containing all detected English text
    """
    # Initialize PaddleOCR with English language model
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang='en',
        use_gpu=False,
        det_db_thresh=0.3,
        det_db_box_thresh=0.5,
        det_db_unclip_ratio=1.8,
        rec_batch_num=6,
        drop_score=0.6
    )
    
    # Get OCR results
    results = ocr.ocr(image_path, cls=True)
    
    # Process all text results
    all_text = []
    if results[0]:
        for line in results[0]:
            text = line[1][0].strip()
            confidence = line[1][1]
            bbox = line[0]
            
            # Skip low confidence or very short results
            if confidence < 0.6 or len(text) < 2:
                continue
            
            # Filter for English characters
            # Simple check if text contains mostly English characters
            non_latin_count = sum(1 for char in text if ord(char) > 127)
            if non_latin_count / len(text) > 0.5:  # Skip if more than 50% non-Latin characters
                continue
                
            all_text.append({
                'text': text,
                'confidence': confidence,
                'bbox': bbox
            })
    
    # Sort results by vertical position (top to bottom, then left to right)
    all_text.sort(key=lambda x: (sum([p[1] for p in x['bbox']]) / 4, sum([p[0] for p in x['bbox']]) / 4))
    
    return all_text

def filter_specific_fields(all_text):
    """
    Filter the extracted text to find specific fields
    
    Args:
        all_text (list): List of dictionaries containing all detected text
        
    Returns:
        dict: Dictionary containing the specific fields
    """
    # Initialize dictionary for required fields
    fields = {
        'form_number': None,
        'receipt_number': None,
        'destination_country': None,
        'transportation_mode': None,
        'total_weight': None,
        'number_of_boxes': None,
        'export_date': None
    }
    
    # Convert all_text to a simple list of text for easier searching
    text_only = [item['text'] for item in all_text]
    
    # Search for each field
    for i, item in enumerate(all_text):
        text = item['text'].lower()
        
        # Form number (P.Q.7)
        if ('form' in text and 'p' in text and 'q' in text) or 'p.q.7' in text.lower():
            match = re.search(r'P\.?Q\.?\s*(\d+)', item['text'], re.IGNORECASE)
            if match:
                fields['form_number'] = f"P.Q.{match.group(1)}"
            else:
                # Look at nearby text items
                for j in range(max(0, i-2), min(len(all_text), i+3)):
                    nearby_text = all_text[j]['text']
                    match = re.search(r'P\.?Q\.?\s*(\d+)', nearby_text, re.IGNORECASE)
                    if match:
                        fields['form_number'] = f"P.Q.{match.group(1)}"
                        break
        
        # Receipt number
        if 'receipt no' in text or 'receipt number' in text:
            # Look for the next item which likely contains the number
            if i + 1 < len(all_text):
                number_text = all_text[i + 1]['text']
                if re.search(r'\d+', number_text):
                    fields['receipt_number'] = number_text
        
        # Check for NP60046795 format receipt number
        if re.search(r'NP\d+', text, re.IGNORECASE):
            fields['receipt_number'] = re.search(r'NP\d+', text, re.IGNORECASE).group()
        
        # Destination country
        if 'destination' in text or 'city and country' in text:
            # Look for country name in nearby text
            for j in range(max(0, i), min(len(all_text), i+3)):
                country_text = all_text[j]['text'].lower()
                if 'china' in country_text:
                    fields['destination_country'] = 'CHINA'
                    break
                elif 'thailand' in country_text:
                    fields['destination_country'] = 'THAILAND'
                    break
                elif any(country in country_text for country in ['japan', 'korea', 'vietnam', 'malaysia']):
                    fields['destination_country'] = all_text[j]['text'].upper()
                    break
        
        # Transportation mode
        if 'means of conveyance' in text or 'transport' in text:
            for j in range(max(0, i), min(len(all_text), i+3)):
                mode_text = all_text[j]['text'].lower()
                if 'road' in mode_text:
                    fields['transportation_mode'] = 'Road transport'
                    break
                elif 'railway' in mode_text or 'rail' in mode_text:
                    fields['transportation_mode'] = 'Railway transport'
                    break
                elif 'truck' in mode_text:
                    fields['transportation_mode'] = 'Truck transport'
                    break
                elif 'ship' in mode_text or 'sea' in mode_text:
                    fields['transportation_mode'] = 'Sea transport'
                    break
        
        # Total weight
        if 'value' in text and not fields['total_weight']:
            # Look for values with numbers and currency
            for j in range(i, min(len(all_text), i+3)):
                value_text = all_text[j]['text']
                if re.search(r'\d+[,\d]*[.\d]+', value_text):
                    fields['total_weight'] = value_text
                    break
        
        # Check for "Baht" or currency values
        if 'baht' in text.lower() or re.search(r'\d+,\d+,\d+', text):
            matches = re.search(r'([\d,\.]+)\s*baht', text.lower())
            if matches:
                fields['total_weight'] = matches.group(1) + ' Baht'
            else:
                numeric_match = re.search(r'\d+[,\d]*[.\d]+', text)
                if numeric_match:
                    fields['total_weight'] = numeric_match.group()
        
        # Number of boxes
        if ('carton' in text.lower() or 'package' in text.lower()) and re.search(r'\d+', text):
            matches = re.search(r'(\d+)\s*[Cc]arton', text)
            if matches:
                fields['number_of_boxes'] = matches.group(1) + ' Carton(s)'
            else:
                # Just extract the number
                numbers = re.findall(r'\d+', text)
                if numbers:
                    fields['number_of_boxes'] = numbers[0] + ' Carton(s)'
        
        # Export date
        if 'date of exportation' in text.lower() or 'date of conveyance' in text.lower():
            for j in range(i, min(len(all_text), i+3)):
                date_text = all_text[j]['text']
                date_match = re.search(r'\d{1,2}/\d{1,2}/\d{2,4}', date_text)
                if date_match:
                    fields['export_date'] = date_match.group()
                    break
    
    # Additional checks for fields that might have been missed
    # Look through all text items for specific patterns
    for item in all_text:
        text = item['text']
        
        # Date formats
        date_match = re.search(r'\d{1,2}/\d{1,2}/\d{2,4}', text)
        if date_match and not fields['export_date']:
            fields['export_date'] = date_match.group()
        
        # Weight in kilograms
        kg_match = re.search(r'(\d+[,\d]*(?:\.\d+)?)\s*kg', text, re.IGNORECASE)
        if kg_match and not fields['total_weight']:
            fields['total_weight'] = kg_match.group(1) + ' kg'
        
        # Value in Baht or other currency
        if not fields['total_weight'] and re.search(r'\d+[,\d]*\.\d+', text):
            # Check if it's in a context that suggests it's a value
            if any(value_word in text.lower() for value_word in ['value', 'amount', 'total', 'baht']):
                fields['total_weight'] = re.search(r'\d+[,\d]*\.\d+', text).group()
    
    return fields

# Main execution
if __name__ == "__main__":
    # Replace with your image path
    image_path = "ex2-large-p1.jpeg"
    
    # Step 1: Extract all text
    all_text = extract_all_english_text(image_path)
    
    # Print all extracted text
    print("=== ALL EXTRACTED TEXT ===")
    for i, item in enumerate(all_text, 1):
        print(f"{i}. {item['text']} (Confidence: {item['confidence']:.2f})")
    
    # Step 2: Filter for specific fields
    extracted_fields = filter_specific_fields(all_text)
    
    # Print filtered fields
    print("\n=== EXTRACTED SPECIFIC FIELDS ===")
    print(f"Form Number: {extracted_fields['form_number'] or 'Not found'}")
    print(f"Receipt Number: {extracted_fields['receipt_number'] or 'Not found'}")
    print(f"Destination Country: {extracted_fields['destination_country'] or 'Not found'}")
    print(f"Transportation Mode: {extracted_fields['transportation_mode'] or 'Not found'}")
    print(f"Total Weight/Value: {extracted_fields['total_weight'] or 'Not found'}")
    print(f"Number of Boxes: {extracted_fields['number_of_boxes'] or 'Not found'}")
    print(f"Export Date: {extracted_fields['export_date'] or 'Not found'}")
    
    # Save results to files
    with open("all_extracted_text.txt", "w", encoding="utf-8") as f:
        f.write("=== ALL EXTRACTED TEXT ===\n\n")
        for i, item in enumerate(all_text, 1):
            f.write(f"{i}. {item['text']} (Confidence: {item['confidence']:.2f})\n")
