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
    if request.file_url.startswith("http"):
        try:
            img_bytes = await url_download(request.file_url)
        except httpx.HTTPStatusError as err:
            raise HTTPException(status_code=err.response.status_code, detail=str(err))
    elif request.file_url.startswith("s3"):
        raise HTTPException(
            status_code=501,
            detail="S3 URL support is not yet implemented",
        )
    else:
        raise HTTPException(
            status_code=422,
            detail="Unsupported file URL scheme. Only 'http(s)' and 's3' are supported.",
        )

    try:
        result = await pq7_pipeline.extract(img_bytes)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))

    return result
