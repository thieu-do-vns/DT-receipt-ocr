from dependency_injector.wiring import inject
import numpy as np
from jaxtyping import UInt8

from dt_receipt_ocr.deps.container import OCRDep, OpenAIDep
from dt_receipt_ocr.models.ocr import PQ7Response, PQ7ModelResponse
from PIL.Image import Image

import cv2
import re

async def extract(img_pil: Image):
    img_np = np.array(img_pil)

    if detect_blur(img_np)[0]:
        return PQ7Response(
            receipt_number = '',
            destination_country = '',
            transportation_mode = '',
            total_weight = '',
            number_of_boxes = 0,
            export_date = '',
            is_blur = True
            )

    enhance_img = enhance_image(img_np)
    ocr_result = _extract_document(img_np)
    # cv2.imwrite('normal_image.jpg', img_np)
    # cv2.imwrite('enhance_image.jpg', enhance_img)
    ocr_text = "# EXTRACTED FIELDS\n"  # Fixed string quote
    for field_name, field_value in ocr_result["region_texts"].items():
        ocr_text += f"{field_name}: {field_value}\n"  # Changed print() to string concatenation, added newline
    total_weight = extract_total_weight(flatten_dict_list(ocr_result["region_texts"]))
    export_date = extract_epxorted_date(ocr_result['region_texts']['middle'])

    print(export_date)

    # Process text with AI
    ai_extraction = await _process_document_with_ai(ocr_text)
    ai_extraction.total_weight = total_weight
    ai_extraction.export_date = export_date

    print(ai_extraction)

    ai_extraction = post_process_ai_response(ai_extraction)

    pq7_response = PQ7Response(**ai_extraction.model_dump())
    return pq7_response

def post_process_ai_response(ai_extraction: PQ7ModelResponse):
    if ('**' in ai_extraction.receipt_number):
        ai_extraction.receipt_number = ""
    if ('by' not in ai_extraction.transportation_mode.lower()):
        ai_extraction.transportation_mode = ""
    country = ai_extraction.destination_country.lower()
    if ("vietnam" not in country) and ("china" not in country) and ("lao" not in country) and ("campuchia") not in country:
        ai_extraction.destination_country = ""
    if re.search(r'NP\d+', ai_extraction.receipt_number, re.IGNORECASE):
            ai_extraction.receipt_number = re.search(r'NP\d+', ai_extraction.receipt_number, re.IGNORECASE).group()
    return ai_extraction

def enhance_image(img_np):
    """
    Preprocess an overexposed image to balance colors and improve readability.
    
    Args:
        image_path (str): Path to the input image
        output_path (str): Path to save the processed image
    """
     
    img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Convert to LAB color space (better for color adjustments)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    # Merge the CLAHE enhanced L-channel back with the a and b channels
    enhanced_lab = cv2.merge((l_clahe, a, b))
    
    # Convert back to RGB color space
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    return enhanced_img
    

def detect_blur(img_np, threshold=100):
    """
    Detect if an image is blurry using the Laplacian variance method.
    
    Args:
        threshold (float): Threshold value to determine blur (lower means more sensitive)
    
    Returns:
        tuple: (is_blurry, laplacian_variance)
    """
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Calculate the Laplacian of the image and compute the variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_variance = laplacian.var()
    
    # Determine if the image is blurry based on the variance
    is_blurry = laplacian_variance < threshold
    
    return is_blurry, laplacian_variance

def extract_epxorted_date(bboxes):
    # Pattern để trích xuất định dạng dd/mm/yyyy
    pattern = r'\d{2}/\d{2}/\d{4}'
    
    # Tìm tất cả các kết quả khớp
    for box in bboxes:
        dates_found = re.findall(pattern, box['text'])
        if dates_found:
            print(f"Tìm thấy {len(dates_found)} ngày tháng:")
            return dates_found[0]
    return ""

def extract_total_weight(bboxes):
    def is_overlap(bbox1, bbox2):
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2

        if (x1 <= x3 <= x2 or x1 <= x4 <= x2) and (y2 < y3 or y4 < y1):
            return True
        return False

    def is_not_second_row(text):
        return ".0000" not in text

    potential = []
    anchor_box = {}
    unit = ""
    for i, bbox in enumerate(bboxes):
        if bbox["text"].lower() == "quantity":
            potential = bboxes[i:]
            anchor_box = bboxes[i]
        if ".0000" in bbox["text"]:
            unit = bbox["text"]

    for bbox in potential:
        if (
            is_overlap(bbox["bbox"], anchor_box["bbox"])
            and bbox["text"] != anchor_box["text"]
            and is_not_second_row(bbox["text"])
        ):
            return "".join(char for char in bbox["text"] if char.isdigit()) + "," + unit
    return ""


def flatten_dict_list(data):
    return [item for k, v in data.items() for item in v]


@inject
async def _process_document_with_ai(document_text, openai_client: OpenAIDep):
    # Make the API call
    
    print("Document text", document_text)
    
    response = await openai_client.beta.chat.completions.parse(
        # model="RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16",
        model = "Qwen3",
        messages=[
            {
            "role": "system",
            "content": """You are a helpful assistant. Extract information EXACTLY as it appears in the provided text, without combining with other texts.
            Return only a single valid JSON object with the shipping details, without any additional text, comments, or trailing content /no_think""",
            },
            {
                "role": "user",
                "content": f"""
        Extract shipping details from the text below and return ONLY a valid JSON object with these fields:
        - "receipt_number": The P.Q.7 receipt number (format NP****)
        - "destination_countries": Country/countries of destination as a single string
        - "transportation_mode": Method of transport
        - "total_weight": Total weight of shipment
        - "number_of_boxes": Number of boxes/cartons
        - "export_date"

        EXTRACTION RULES:
        1. For receipt_number: Look for number with format NP****
        2. For destination_countries: Find text near "City and country of destination"
        - Return as a SINGLE STRING, preserving the EXACT original format
        - Include all geographical details (provinces, cities, etc.) exactly as written
        - **If multiple countries are listed, include them all in the same string (e.g., "Youyiguan CHINA, Quang Binh VIENAM, Thakhek LAO PEOPLE")**
        - **Correct misspelled country names (e.g., "CHNA" → "CHINA")**
        - Destination country should not be Thailand
        - Ignore phrases like "IMPORT AND EXPORT TRADE"
        3. For transportation_mode: Look for phrases starting with "by" (e.g., "By Train", "By Truck", "By Truck and Railway")
        4. For number_of_boxes: Look for numeric values, may include unit "CARTONS" or "cartons"
        5. For export_date: For export_date: Find date near phrase "Date of exportation" and in middle or bottom keyword part. It have format dd/mm/yyyy
        6. **If you can not find any suitable information, let this field empty**
        Return ONLY the JSON object without additional text, comments, or explanations.

        TEXT TO PROCESS:
        {document_text}
        """,
            },
        ],
        temperature=0.2,
        max_tokens=3096,
        response_format=PQ7ModelResponse
    )

    # Return the AI response
    return response.choices[0].message.parsed


def _extract_document(img_np: UInt8):
    result = {"status": "success", "fields": {}, "region_texts": {}, "raw_text": []}

    region_texts = _extract_fields_by_region_wrapper(img_np)
    result["region_texts"] = region_texts
    # Combine all text for raw_text
    all_text = []
    for texts in region_texts.values():
        all_text.extend(texts)
    result["raw_text"] = all_text

    return result


def _extract_fields_by_region_wrapper(img_np: UInt8):
    # Extract regions from the image
    regions = _extract_regions_from_image(img_np)

    # Extract text from each region
    region_texts = {}
    for region_name, region_image in regions.items():
        region_texts[region_name] = _extract_text_from_region(region_image, region_name)

    return region_texts


def _extract_regions_from_image(img_np):
    # Get image dimensions
    height, width = img_np.shape[:2]

    # Define regions (based on typical Phytosanitary Certificate layout)
    # Format: [x_start, y_start, x_end, y_end]
    regions = {
        # Upper right corner for Form P.Q.7 and receipt number
        "upper_right": [int(width * 0.5), 0, width, int(height * 0.3)],
        # Middle section for destination and transportation
        "middle": [0, int(height * 0.3), width, int(height * 0.7)],
        # Bottom section for weight, boxes, and export date
        "bottom": [0, int(height * 0.6), width, height],
    }

    # Extract each region
    region_images = {}
    for region_name, coords in regions.items():
        x_start, y_start, x_end, y_end = coords
        region_images[region_name] = img_np[y_start:y_end, x_start:x_end].copy()

        # # Save region for debugging (optional)
        cv2.imwrite(f"region_{region_name}_api.jpg", region_images[region_name])

    return region_images


@inject
def _extract_text_from_region(region_img_np, region_name, ocr: OCRDep):

    temp_region_path = f"temp_{region_name}_api.jpg"
    # cv2.imwrite(temp_region_path, region_img_np)
  
    results = ocr.ocr(region_img_np[...,::-1], cls=True)
    height, width = region_img_np.shape[:2]

    # Process results
    region_text = []
    if results[0]:
        for bbox, (text, confidence) in results[0]:
            text = text.strip()

            if region_name == "upper_right":
                # update min_x bbox[0] -> (min_x, min_y)
                bbox[0][0] = bbox[0][0] + int(width * 0.5)
                bbox[1][0] = bbox[1][0] + int(width * 0.5)
                bbox[2][0] = bbox[2][0] + int(width * 0.5)
                bbox[3][0] = bbox[3][0] + int(width * 0.5)
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
            if (
                non_latin_count / len(text) > 0.5
            ):  # Skip if more than 50% non-Latin characters
                continue

            region_text.append({"text": text, "bbox": bbox})

    # Sort results by position (top to bottom, then left to right)
    region_text.sort(
        key=lambda x: (
            sum([p[1] for p in x["bbox"]]) / 4,
            sum([p[0] for p in x["bbox"]]) / 4,
        )
    )

    return region_text
