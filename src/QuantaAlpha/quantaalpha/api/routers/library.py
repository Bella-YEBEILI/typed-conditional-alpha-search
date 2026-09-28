from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from quantaalpha.api.deps import get_library_service
from quantaalpha.api.schemas.library import (
    FactorDetail,
    FactorListResponse,
    FactorUpdateRequest,
)
from quantaalpha.api.services.library_service import LibraryService

router = APIRouter(prefix="/api/library", tags=["library"])


@router.get("/list")
def list_libraries(lib_svc: LibraryService = Depends(get_library_service)):
    return {"available": lib_svc.list_libraries()}


@router.get("/factors", response_model=FactorListResponse)
def get_factors(
    suffix: str | None = Query(None),
    quality: str = Query("all"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    sort_by: str = Query("added_at"),
    sort_order: str = Query("desc"),
    search: str = Query(""),
    lib_svc: LibraryService = Depends(get_library_service),
):
    return lib_svc.get_factors(
        suffix=suffix,
        quality=quality,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
        search=search,
    )


@router.get("/factors/{factor_id}", response_model=FactorDetail)
def get_factor_detail(
    factor_id: str,
    suffix: str | None = Query(None),
    lib_svc: LibraryService = Depends(get_library_service),
):
    detail = lib_svc.get_factor_detail(factor_id, suffix)
    if not detail:
        raise HTTPException(status_code=404, detail="Factor not found")
    return detail


@router.put("/factors/{factor_id}")
def update_factor(
    factor_id: str,
    body: FactorUpdateRequest,
    suffix: str | None = Query(None),
    lib_svc: LibraryService = Depends(get_library_service),
):
    result = lib_svc.update_factor_expression(
        factor_id, body.factor_expression, suffix,
        description=body.factor_description,
        formulation=body.factor_formulation,
    )
    if not result:
        raise HTTPException(status_code=404, detail="Factor not found")
    return {"factor_id": factor_id, "updated": True, "detail": result}


@router.delete("/factors/{factor_id}")
def delete_factor(
    factor_id: str,
    suffix: str | None = Query(None),
    lib_svc: LibraryService = Depends(get_library_service),
):
    ok = lib_svc.delete_factor(factor_id, suffix)
    if not ok:
        raise HTTPException(status_code=404, detail="Factor not found")
    return {"factor_id": factor_id, "deleted": True}
