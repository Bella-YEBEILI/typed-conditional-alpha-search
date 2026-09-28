from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from quantaalpha.api.config import get_settings
from quantaalpha.api.routers import dashboard, library, mining, backtest, analysis
from quantaalpha.api.ws import mining_ws, backtest_ws


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="QuantaAlpha API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(dashboard.router)
    app.include_router(library.router)
    app.include_router(mining.router)
    app.include_router(backtest.router)
    app.include_router(analysis.router)
    app.include_router(mining_ws.router)
    app.include_router(backtest_ws.router)

    results_dir = settings.project_root / settings.static_results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static/results", StaticFiles(directory=str(results_dir)), name="results")

    @app.get("/api/health")
    def health_check():
        return {"status": "ok", "version": "0.1.0"}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "quantaalpha.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
