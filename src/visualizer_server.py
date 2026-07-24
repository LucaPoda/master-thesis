# visualizer_server.py
import threading
import asyncio
import json
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from agent_state import GraphTracker

class GraphVisualizerBridge:
    def __init__(self, tracker: GraphTracker, config_colors: dict, host: str = "127.0.0.1", port: int = 8000):
        self.tracker = tracker
        self.config_colors = config_colors
        self.host = host
        self.port = port
        
        self.app = FastAPI()
        self.active_connections: list[WebSocket] = []
        self.loop: asyncio.AbstractEventLoop | None = None
        
        self._setup_routes()
        # Subscribe to the tracker's event
        self.tracker.add_callback(self.on_graph_updated)

    def _setup_routes(self):
        @self.app.get("/")
        async def get():
            with open("src/index.html", "r", encoding="utf-8") as f:
                return HTMLResponse(f.read())

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.active_connections.append(websocket)
            # Send the initial state just after connecting
            await websocket.send_text(self._serialize_graph())
            try:
                while True:
                    await websocket.receive_text() # Mantieni aperta la connessione
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)

        @self.app.on_event("startup")
        async def startup_event():
            # Capture the event loop of FastAPI for use in other threads
            self.loop = asyncio.get_running_loop()

    def start(self):
        """Start the web server in a daemon thread."""
        def run_server():
            uvicorn.run(self.app, host=self.host, port=self.port, log_level="error")
            
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()

    def _serialize_graph(self) -> str:
        data = self.tracker.get_graph_data()
        data["colors"] = self.config_colors 
        return json.dumps(data)

    def on_graph_updated(self):
        """Callback called from the main simulation thread."""
        if not self.loop or not self.active_connections:
            return

        # Plan the asynchronous sending in a thread-safe manner
        payload = self._serialize_graph()
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), self.loop)

    async def _broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass