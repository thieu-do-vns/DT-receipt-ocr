from fastapi import FastAPI
import hydra
import dt_receipt_ocr
from dt_receipt_ocr.deps import Container
from dt_receipt_ocr.routers.v1 import ocr
from contextlib import asynccontextmanager
import dt_receipt_ocr.core.runner
from omegaconf import OmegaConf


@asynccontextmanager
async def lifespan(app: FastAPI):
    container = Container()
    with hydra.initialize(version_base=None, config_path="conf"):
        cfg = hydra.compose(config_name="main")
    container.cfg.from_dict(OmegaConf.to_object(cfg))
    container.wire(modules=[ocr, dt_receipt_ocr.core.runner])
    await container.init_resources()

    yield

    await container.shutdown_resources()
    container.unwire()


api = FastAPI(lifespan=lifespan)
api.include_router(ocr.router, prefix="/v1")
