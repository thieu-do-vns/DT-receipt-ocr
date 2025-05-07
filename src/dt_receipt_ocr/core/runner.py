import cv2
from dependency_injector.wiring import inject
import numpy as np
from jaxtyping import UInt8

from dt_receipt_ocr.deps.container import OCRDep


def extract_pq7(img_bytes: bytes):
    img_np = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    ocr_result = extract_document(img_np)
    ocr_text = "# EXTRACTED FIELDS\n"  # Fixed string quote
    for field_name, field_value in ocr_result["region_texts"].items():
        ocr_text += f"{field_name}: {field_value}\n"  # Changed print() to string concatenation, added newline
    print(ocr_text)


def extract_document(img_np: UInt8):
    result = {"status": "success", "fields": {}, "region_texts": {}, "raw_text": []}

    region_texts = extract_fields_by_region_wrapper(img_np)
    result["region_texts"] = region_texts
    # Combine all text for raw_text
    all_text = []
    for texts in region_texts.values():
        all_text.extend(texts)
    result["raw_text"] = all_text

    return result


def extract_fields_by_region_wrapper(img_np: UInt8):
    # Extract regions from the image
    regions = extract_regions_from_image(img_np)

    # Extract text from each region
    region_texts = {}
    for region_name, region_image in regions.items():
        region_texts[region_name] = extract_text_from_region(region_image, region_name)

    return region_texts


def extract_regions_from_image(img_np):
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
def extract_text_from_region(region_img_np, region_name, ocr: OCRDep):
    results = ocr.ocr(region_img_np, cls=True)

    # Process results
    region_text = []
    if results[0]:
        for line in results[0]:
            text = line[1][0].strip()
            confidence = line[1][1]
            bbox = line[0]

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
