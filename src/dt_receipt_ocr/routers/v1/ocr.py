from fastapi import APIRouter, HTTPException
import httpx
from dt_receipt_ocr.core import pq7_pipeline
from dt_receipt_ocr.models import PQ7Response, PQ7Request
from pydantic import HttpUrl
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
    except httpx.HTTPStatusError as err:
        raise HTTPException(status_code=err.response.status_code, detail=str(err))

    result = await pq7_pipeline.extract(img_bytes)
    return result
