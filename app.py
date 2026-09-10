from pathlib import Path
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from backend import (
    run_travel_agent,
    resume_travel_agent,
)

import nest_asyncio

nest_asyncio.apply()

BASE_DIR = Path(__file__).resolve().parent

# =========================================================
# FastAPI application
# =========================================================

app = FastAPI(
    title="TravelMind AI",
    description=(
        "Personalized, practical multi-agent travel planning "
        "with LangGraph, MCP, travel integrations, PostgreSQL, "
        "Supervisor, Guardrails, and Human-in-the-Loop"
    ),
    version="2.0.0",
)

app.mount(
    "/static",
    StaticFiles(
        directory=str(BASE_DIR / "static")
    ),
    name="static",
)

templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)

# =========================================================
# Request models
# =========================================================

class TravelRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ApprovalRequest(BaseModel):
    thread_id: str = Field(
        min_length=1
    )
    approved: bool
    feedback: str = ""

# =========================================================
# Home
# =========================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )

# =========================================================
# Travel planning
# =========================================================

@app.post("/api/travel")
async def travel_planner(
    request_data: TravelRequest,
):
    try:
        user_message = request_data.message.strip()

        if not user_message:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Message cannot be empty.",
                },
            )

        result = run_travel_agent(
            user_input=user_message,
            thread_id=request_data.thread_id,
        )

        return JSONResponse(
            content={
                "success": True,
                **result,
            }
        )

    except Exception as exc:
        print(
            "ERROR:",
            exc,
            flush=True,
        )
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(exc),
            },
        )

# =========================================================
# Human-in-the-Loop approval
# =========================================================

@app.post("/api/travel/approve")
async def approve_travel_plan(
    request_data: ApprovalRequest,
):
    try:
        if (
            not request_data.approved
            and not request_data.feedback.strip()
        ):
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": (
                        "Please provide revision feedback "
                        "when rejecting the draft."
                    ),
                },
            )

        result = resume_travel_agent(
            thread_id=request_data.thread_id,
            approved=request_data.approved,
            feedback=request_data.feedback,
        )

        return JSONResponse(
            content={
                "success": True,
                **result,
            }
        )

    except Exception as exc:
        print(
            "APPROVAL ERROR:",
            exc,
            flush=True,
        )
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(exc),
            },
        )

# =========================================================
# Health check
# =========================================================

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "TravelMind AI API is running",
        "features": [
            "supervisor_agent",
            "input_guardrail",
            "human_in_the_loop",
        ],
    }

# =========================================================
# Favicon
# =========================================================

@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(
        content={}
    )

# =========================================================
# Application entry point
# =========================================================

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )