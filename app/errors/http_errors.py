class APIError(Exception):
    status_code: int = 500
    error_type: str = "APIError"

    def __init__(self, detail: str):
        self.detail = detail

    def to_dict(self):
        return {
            "type": self.error_type,
            "status": self.status_code,
            "detail": self.detail,
        }
    
class ValidationError(APIError):
    status_code = 400
    error_type = "ValidationError"

class InternalError(APIError):
    status_code = 500
    error_type = "InternalError"

class NotFoundError(APIError):
    status_code = 404
    error_type = "NotFoundError"

class ModelLoadError(APIError):
    status_code = 500
    error_type = "ModelLoadError"

class UnprocessableEntityError(APIError):
    status_code = 422
    error_type = "UnprocessableEntityError"