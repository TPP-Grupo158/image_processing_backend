from fastapi.responses import JSONResponse
from fastapi import Request
from http_errors import APIError


def register_exception_handlers(app):
    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(),
        )