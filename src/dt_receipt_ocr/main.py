from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader, APIKey
import hydra
import dt_receipt_ocr.core.pq7_pipeline
from dt_receipt_ocr.deps import Container
from dt_receipt_ocr.routers.v1 import ocr
from contextlib import asynccontextmanager
from omegaconf import OmegaConf

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

config_container = {}

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if not api_key_header:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="API Key header is missing"
        )
    if api_key_header == config_container.get("api_key"):
        return api_key_header
    
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key"
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    container = Container()
    with hydra.initialize(version_base=None, config_path="conf"):
        cfg = hydra.compose(config_name="main")
    if "security" in cfg and "api_key" in cfg.security:
        config_container["api_key"] = cfg.security.api_key
    else:
        # API key mặc định nếu không có trong cấu hình
        config_container["api_key"] = "default-api-key"


    container.cfg.from_dict(OmegaConf.to_object(cfg))
    container.wire(modules=[ocr, dt_receipt_ocr.core.pq7_pipeline])
    await container.init_resources()

    yield

    await container.shutdown_resources()
    container.unwire()


api = FastAPI(lifespan=lifespan)
api.include_router(
    ocr.router, 
    prefix="/dt", 
    dependencies=[Depends(get_api_key)]
)
