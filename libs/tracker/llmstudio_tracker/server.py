from threading import Event, Thread

import requests
import uvicorn
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from llmstudio_tracker.config import TRACKING_HOST, TRACKING_PORT
from llmstudio_tracker.logs.endpoints import LogsRoutes
from llmstudio_tracker.prompt_manager.endpoints import PromptsRoutes
from llmstudio_tracker.session.endpoints import SessionsRoutes
from llmstudio_tracker.utils import get_current_version

TRACKING_HEALTH_ENDPOINT = "/health"
TRACKING_TITLE = "LLMstudio Tracker API"
TRACKING_DESCRIPTION = "The tracking API for LLM interactions"
TRACKING_VERSION = get_current_version()
TRACKING_BASE_ENDPOINT = "/api/tracking"

_tracker_server_started = False


## Tracking
def create_tracking_app(started_event: Event) -> FastAPI:
    app = FastAPI(
        title=TRACKING_TITLE,
        description=TRACKING_DESCRIPTION,
        version=TRACKING_VERSION,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get(TRACKING_HEALTH_ENDPOINT)
    def health_check():
        """Health check endpoint to ensure the API is running."""
        return {"status": "healthy", "message": "Tracking is up and running"}

    tracking_router = APIRouter(prefix=TRACKING_BASE_ENDPOINT)
    LogsRoutes(tracking_router)
    SessionsRoutes(tracking_router)
    PromptsRoutes(tracking_router)

    app.include_router(tracking_router)

    @app.on_event("startup")
    async def startup_event():
        started_event.set()
        print(f"Running LLMstudio Tracking on http://{TRACKING_HOST}:{TRACKING_PORT} ")

    return app


def run_tracker_app(started_event: Event):
    try:
        tracking = create_tracking_app(started_event)
        uvicorn.run(
            tracking,
            host=TRACKING_HOST,
            port=TRACKING_PORT,
            log_level="warning",
        )
    except Exception as e:
        print(f"Error running LLMstudio Tracking: {e}")


def is_server_running(url, path="/health"):
    try:
        response = requests.get(f"{url}{path}")
        if response.status_code == 200 and response.json().get("status") == "healthy":
            return True
    except requests.ConnectionError:
        pass
    return False


def start_server_component(host, port, run_func, server_name):
    if not is_server_running(url=f"http://{host}:{port}"):
        started_event = Event()
        thread = Thread(target=run_func, daemon=True, args=(started_event,))
        thread.start()
        started_event.wait()  # wait for startup, this assumes the event is set somewhere
        return thread
    else:
        print(f"{server_name} server already running on {host}:{port}")
        return None


def setup_tracking_server():
    global _tracker_server_started
    tracker_thread = None
    if not _tracker_server_started:
        tracker_thread = start_server_component(
            TRACKING_HOST, TRACKING_PORT, run_tracker_app, "Tracker"
        )
        _tracker_server_started = True
    return tracker_thread
