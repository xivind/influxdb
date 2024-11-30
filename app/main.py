#!/usr/bin/env python3

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from app import chat

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="app/templates")

@app.get("/")
async def root(request: Request):
    """Endpoint to get landing page"""
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/chat")
async def chat_endpoint(request: Request):
    """Endpoint to handle chat submissions"""
    data = await request.json()
    query = data.get("query")
    
    async def event_generator():
        async for token in chat.get_streaming_response(query):
            yield f"data: {token}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream")
