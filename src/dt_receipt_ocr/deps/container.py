import httpx
from typing import Annotated

from dependency_injector import containers, providers
from dependency_injector.wiring import Provide
from paddleocr import PaddleOCR
from openai import AsyncOpenAI


async def init_http_client():
    async with httpx.AsyncClient() as client:
        yield client


class Container(containers.DeclarativeContainer):
    cfg = providers.Configuration()
    ocr = providers.Singleton(
        PaddleOCR,
        use_angle_cls=True,
        lang="en",
        use_gpu=False,
        det_db_thresh=0.3,
        det_db_box_thresh=0.5,
        det_db_unclip_ratio=1.8,
        rec_batch_num=6,
        drop_score=0.6,
    )
    http_client = providers.Resource(init_http_client)
    openai_client = providers.Singleton(AsyncOpenAI, base_url=cfg.bas)


HttpClientDep = Annotated[httpx.AsyncClient, Provide[Container.http_client]]
OCRDep = Annotated[PaddleOCR, Provide[Container.ocr]]
OpenAIDep = Annotated[AsyncOpenAI, Provide[Container.openai_client]]
