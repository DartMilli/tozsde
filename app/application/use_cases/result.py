from typing import Any, Dict, TypedDict


class UseCaseError(TypedDict):
    message: str


class UseCaseResult(TypedDict, total=False):
    status: str
    use_case: str
    data: Any
    meta: Dict[str, Any]
    error: UseCaseError


def ok(use_case: str, data: Any = None, **meta) -> UseCaseResult:
    return {
        "status": "ok",
        "use_case": use_case,
        "data": data,
        "meta": meta or {},
    }


def error(use_case: str, message: str, **meta) -> UseCaseResult:
    return {
        "status": "error",
        "use_case": use_case,
        "data": None,
        "error": {"message": message},
        "meta": meta or {},
    }
