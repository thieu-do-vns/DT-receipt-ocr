import cv2
from dependency_injector.wiring import inject
import numpy as np
from jaxtyping import UInt8

from dt_receipt_ocr.deps.container import OCRDep, OpenAIDep
from dt_receipt_ocr.models.ocr import PQ7Response


async def extract(img_bytes: bytes):
    img_np = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    ocr_result = _extract_document(img_np)
    ocr_text = "# EXTRACTED FIELDS\n"  # Fixed string quote
    for field_name, field_value in ocr_result["region_texts"].items():
        ocr_text += f"{field_name}: {field_value}\n"  # Changed print() to string concatenation, added newline

    # Process text with AI
    ai_extraction = await _process_document_with_ai(ocr_text)
    return ai_extraction


@inject
async def _process_document_with_ai(document_text, openai_client: OpenAIDep):
    # Make the API call
    response = await openai_client.beta.chat.completions.parse(
        model="RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16",
        messages=[
            {
                "role": "system",
                "content": """You are a helpful assistant. Extract information EXACTLY as it appears in the provided text, without combining with other texts.
        Return only a single valid JSON object with the shipping details, without any additional text, comments, or trailing content""",
            },
            {
                "role": "user",
                "content": f"""
        Extract ONLY the following fields from the text below and return in json format:
        - P.Q.7 receipt number
        - Destination country
        - Transportation mode
        - Total weight
        - Number of boxes
        - Export date

        CONTEXT:
        1. Process ONLY the text provided in this single request
        2. For destination country, preserve the EXACT text format - do not separate or modify country names
        For example, if text contains 'Youyiguan CHINAQuang Binh VIETNAM', return 'Youyiguan CHINA, Quang Binh VIETNAM'
        Do not consider phrases like 'IMPORT AND EXPORT TRADE' as destination countries
        3. Number of boxes maybe have unit of CARTONS (cartons)
        4. Export date should be near to phrase: Date of exportation and have format dd/mm/yyyy

        Return only a single valid JSON object with the shipping details, without any additional text, comments, or trailing content

        Text to process:
        {document_text}
        """,
            },
        ],
        temperature=0.2,
        max_tokens=2048,
        response_format=PQ7Response,
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
        "upper_right": [int(width * 0.6), 0, width, int(height * 0.3)],
        # Middle section for destination and transportation
        "middle": [0, int(height * 0.3), width, int(height * 0.7)],
        # Bottom section for weight, boxes, and export date
        "bottom": [0, int(height * 0.7), width, height],
    }

    # Extract each region
    region_images = {}
    for region_name, coords in regions.items():
        x_start, y_start, x_end, y_end = coords
        region_images[region_name] = img_np[y_start:y_end, x_start:x_end].copy()

        # # Save region for debugging (optional)
        # cv2.imwrite(f"region_{region_name}.jpg", region_images[region_name])

    return region_images


@inject
def _extract_text_from_region(region_img_np, region_name, ocr: OCRDep):
    results = ocr.ocr(region_img_np, cls=True)

    height, width = region_img_np.shape[:2]

    # Process results
    region_text = []
    if results[0]:
        for bbox, (text, confidence) in results[0]:
            text = text.strip()

            if region_name == "upper_right":
                # update min_x bbox[0] -> (min_x, min_y)
                bbox[0][0] = bbox[0][0] + int(width * 0.6)
                bbox[1][0] = bbox[1][0] + int(width * 0.6)
                bbox[2][0] = bbox[2][0] + int(width * 0.6)
                bbox[3][0] = bbox[3][0] + int(width * 0.6)
            if region_name == "middle":
                # update min_y
                bbox[0][1] = bbox[0][1] + int(height * 0.3)
                bbox[1][1] = bbox[1][1] + int(height * 0.3)
                bbox[2][1] = bbox[2][1] + int(height * 0.3)
                bbox[3][1] = bbox[3][1] + int(height * 0.3)

            if region_name == "bottom":
                # update y
                bbox[0][1] = bbox[0][1] + int(height * 0.7)
                bbox[1][1] = bbox[1][1] + int(height * 0.7)
                bbox[2][1] = bbox[2][1] + int(height * 0.7)
                bbox[3][1] = bbox[3][1] + int(height * 0.7)

            # Skip low confidence or very short results
            if confidence < 0.6 or len(text) < 2:
                continue

            # Filter for English characters
            non_latin_count = sum(1 for char in text if ord(char) > 127)
            if (
                non_latin_count / len(text) > 0.5
            ):  # Skip if more than 50% non-Latin characters
                continue

            region_text.append({"text": text, "confidence": confidence, "bbox": bbox})

    # Sort results by position (top to bottom, then left to right)
    region_text.sort(
        key=lambda x: (
            sum([p[1] for p in x["bbox"]]) / 4,
            sum([p[0] for p in x["bbox"]]) / 4,
        )
    )

    return region_text
