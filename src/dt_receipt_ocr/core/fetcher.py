from pydantic import HttpUrl

async def url_download(image_url: HttpUrl):
    response = await http_client.get(str(image_url))
    response.raise_for_status()
    return response.content
