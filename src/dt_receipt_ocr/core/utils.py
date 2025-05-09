from dt_receipt_ocr.models.ocr import PQ7Response, PQ7ModelResponse

def is_missing_field_pq7_response(pq7_response: PQ7ModelResponse) -> bool:
    """
    Check if more than 50% of fields in PQ7ModelResponse are missing.
    Missing is defined as:
    - String fields: empty string ('')
    - Integer fields: 0
    
    Args:
        pq7_response: A PQ7ModelResponse instance
        
    Returns:
        bool: True if more than 50% of fields are missing, False otherwise
    """

    all_fields = pq7_response.model_fields
    total_fields = len(all_fields)
    
    # Count missing fields
    missing_count = 0
    for field_name, field_info in all_fields.items():
        value = getattr(pq7_response, field_name, None)
        field_type = field_info.annotation
        
        # Check if field is missing based on its type
        if field_type is str and value == '':
            missing_count += 1
        elif field_type is int and value == 0:
            missing_count += 1
    
    # Calculate percentage of missing fields
    missing_percentage = (missing_count / total_fields) * 100
    
    # Return True if more than 50% fields are missing
    return missing_percentage > 50