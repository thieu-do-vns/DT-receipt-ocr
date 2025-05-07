from fastapi import APIRouter, HTTPException
from dt_receipt_ocr.core.runner import extract_pq7
from dt_receipt_ocr.models import PQ7Response, PQ7Request
from jaxtyping import UInt8
from pydantic import HttpUrl
import cv2
from dt_receipt_ocr.deps import HttpClientDep
from dependency_injector.wiring import inject


router = APIRouter()


@inject
async def url_download(image_url: HttpUrl, http_client: HttpClientDep):
    response = await http_client.get(str(image_url))
    response.raise_for_status()
    return response.content


@router.post("/ocr_pq7")
async def ocr_pq7(request: PQ7Request) -> PQ7Response:
    try:
        img_bytes = await url_download(request.file_url)
    except Exception as err:
        raise HTTPException(status_code=404, detail=f"{type(err)}: {str(err)}")

    result = extract_pq7(img_bytes)

    return PQ7Response()
