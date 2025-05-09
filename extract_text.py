import os
import cv2
import numpy as np
from paddleocr import PaddleOCR
import re
import concurrent.futures
from image_utils import preprocess_overexposed_image

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

def extract_all_english_text(image_path):
    """
    Extract all English text from the document using PaddleOCR
    
    Args:
        image_path (str): Path to the document image
        
    Returns:
        list: List of dictionaries containing all detected English text
    """

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
        
        # Check for "kg" or currency values
        if 'kg' in text.lower() or re.search(r'\d+,\d+,\d+', text):
            matches = re.search(r'([\d,\.]+)\s*kg', text.lower())
            if matches:
                fields['total_weight'] = matches.group(1) + ' kg'
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
        
        # Value in kg
        if not fields['total_weight'] and re.search(r'\d+[,\d]*\.\d+', text):
            # Check if it's in a context that suggests it's a value
            if any(value_word in text.lower() for value_word in ['value', 'amount', 'total', 'kg']):
                fields['total_weight'] = re.search(r'\d+[,\d]*\.\d+', text).group()
    
    return fields

def extract_regions_from_image(image_path):
    """
    Divide the image into specific regions for targeted OCR

    Args:
        image_path (str): Path to the document image

    Returns:
        dict: Dictionary containing region images
    """
    # Read the original image
    image = cv2.imread(image_path)
    # image = preprocess_overexposed_image(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    height, width = image.shape[:2]

    # Define regions (based on typical Phytosanitary Certificate layout)
    # Format: [x_start, y_start, x_end, y_end]
    regions = {
        # Upper right corner for Form P.Q.7 and receipt number
        'upper_right': [int(width * 0.3), 0, width, int(height * 0.3)],

        # Middle section for destination and transportation
        'middle': [0, int(height * 0.3), width, int(height * 0.7)],

        # Bottom section for weight, boxes, and export date
        'bottom': [0, int(height * 0.6), width, height]
    }

    # Extract each region
    region_images = {}
    for region_name, coords in regions.items():
        x_start, y_start, x_end, y_end = coords
        region_images[region_name] = image[y_start:y_end, x_start:x_end].copy()

        # # Save region for debugging (optional)
        cv2.imwrite(f"region_{region_name}.jpg", region_images[region_name])

    return region_images

def extract_text_from_region(region_image, region_name):
    """
    Extract text from a specific region using PaddleOCR

    Args:
        region_image: Image of the region
        region_name (str): Name of the region for optimization

    Returns:
        list: List of text items found in the region
    """


    # Save region to a temporary file
    temp_region_path = f"temp_{region_name}.jpg"
    cv2.imwrite(temp_region_path, region_image)

    # Get OCR results
    results = ocr.ocr(region_image, cls=True)
    height, width = region_image.shape[:2]

    # Process results
    region_text = []
    if results[0]:
        for line in results[0]:
            text = line[1][0].strip()
            confidence = line[1][1]
            bbox = line[0]
            
            if region_name == "upper_right":
              # update min_x bbox[0] -> (min_x, min_y)
              bbox[0][0] = bbox[0][0] + int(width * 0.3)
              bbox[1][0] = bbox[1][0] + int(width * 0.3)
              bbox[2][0] = bbox[2][0] + int(width * 0.3)
              bbox[3][0] = bbox[3][0] + int(width * 0.3)
            if region_name == "middle":
              # update min_y 
              bbox[0][1] = bbox[0][1] + int(height * 0.3)
              bbox[1][1] = bbox[1][1] + int(height * 0.3)
              bbox[2][1] = bbox[2][1] + int(height * 0.3)
              bbox[3][1] = bbox[3][1] + int(height * 0.3)
              
            if region_name == "bottom":
              # update y
              bbox[0][1] = bbox[0][1] + int(height * 0.6)
              bbox[1][1] = bbox[1][1] + int(height * 0.6)
              bbox[2][1] = bbox[2][1] + int(height * 0.6)
              bbox[3][1] = bbox[3][1] + int(height * 0.6)

            # Skip low confidence or very short results
            if confidence < 0.6 or len(text) < 2:
                continue

            # Filter for English characters
            non_latin_count = sum(1 for char in text if ord(char) > 127)
            if non_latin_count / len(text) > 0.5:  # Skip if more than 50% non-Latin characters
                continue

            region_text.append({
                'text': text,
                'confidence': confidence,
                'bbox': bbox
            })

    # Clean up temporary file
    if os.path.exists(temp_region_path):
        # os.remove(temp_region_path)
        pass

    # Sort results by position (top to bottom, then left to right)
    region_text.sort(key=lambda x: (sum([p[1] for p in x['bbox']]) / 4, sum([p[0] for p in x['bbox']]) / 4))

    return region_text

def extract_fields_by_region(image_path):
    """
    Extract specific fields from document by analyzing separate regions

    Args:
        image_path (str): Path to the document image

    Returns:
        dict: Dictionary containing all extracted fields
    """
    # Extract regions from the image
    regions = extract_regions_from_image(image_path)

    # Extract text from each region
    region_texts = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Create a dictionary mapping each future to its region name
        future_to_region = {
            executor.submit(extract_text_from_region, region_image, region_name): region_name
            for region_name, region_image in regions.items()
        }
        
        # Process the results as they complete
        for future in concurrent.futures.as_completed(future_to_region):
            region_name = future_to_region[future]
            try:
                region_texts[region_name] = future.result()
            except Exception as e:
                print(f"Error processing region {region_name}: {e}")
                region_texts[region_name] = f"ERROR: {str(e)}"
    
    return region_texts

# Main execution
if __name__ == "__main__":
    # Replace with your image path
    # image_path = "test_image/image5/ex5-large-p1.jpeg"
    image_path = "rotate.jpg"
    
    # Extract fields by region
    region_texts = extract_fields_by_region(image_path)

    # Print all text by region
    print("=== EXTRACTED TEXT BY REGION ===")
    for region_name, texts in region_texts.items():
        print(f"\n-- {region_name.upper()} REGION --")
        for i, item in enumerate(texts, 1):
            print(f"{i}. {item['text']} (Confidence: {item['confidence']:.2f})")
    