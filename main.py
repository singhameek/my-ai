import os
import json
import time
import asyncio
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import google.generativeai as genai
import httpx

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
MACRODROID_WEBHOOK_URL = os.environ.get("MACRODROID_WEBHOOK_URL")

genai.configure(api_key=GEMINI_API_KEY)

router_model = genai.GenerativeModel(
    "gemini-2.5-flash",
    generation_config={"response_mime_type": "application/json"}
)
deep_model = genai.GenerativeModel("gemini-2.5-pro")

app = FastAPI()

# ---- In-memory state (swap for DB later) ----
device_registry = {}
conversation_log = {}

# ---- Models ----
class ProcessRequest(BaseModel):
    prompt: str
    device_id: str
    device_type: str

class DeviceContext(BaseModel):
    device_id: str
    device_type: str
    capabilities: list[str] = []

ROUTER_PROMPT = """Output ONLY JSON, no markdown, no prose:
{"route": "local"|"direct"|"deep", "reason": str, "webhook_action": str|null, "answer": str|null}

route=local -> prompt maps to a known device automation command
route=direct -> short factual/conversational answer, put it in "answer"
route=deep -> needs multi-step reasoning, code, or long-form output
"""

async def safe_generate(model, prompt, retries=2):
    for i in range(retries):
        try:
            return await asyncio.to_thread(model.generate_content, prompt)
        except Exception as e:
            if i == retries - 1:
                raise HTTPException(status_code=503, detail=f"Gemini error: {e}")
            await asyncio.sleep(2 ** i)

async def route_prompt(prompt: str, ctx: DeviceContext) -> dict:
    full_prompt = f"{ROUTER_PROMPT}\nDevice: {ctx.device_type}\nCapabilities: {ctx.capabilities}\nPrompt: {prompt}"
    resp = await safe_generate(router_model, full_prompt)
    try:
        return json.loads(resp.text)
    except json.JSONDecodeError:
        return {"route": "direct", "answer": resp.text}

async def fire_webhook(url: str, payload, label: str):
    if not url:
        return
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            data = payload if isinstance(payload, dict) else {"content": str(payload)}
            await client.post(url, json=data)
    except httpx.HTTPError as e:
        print(f"[webhook:{label}] failed: {e}")

@app.post("/api/process")
async def process(req: ProcessRequest, background_tasks: BackgroundTasks):
    ctx = device_registry.get(req.device_id) or DeviceContext(
        device_id=req.device_id, device_type=req.device_type
    )
    device_registry[req.device_id] = ctx

    routing = await route_prompt(req.prompt, ctx)
    route = routing.get("route", "direct")

    if route == "local":
        action = routing.get("webhook_action")
        result = {"source": "local", "content": action}
        background_tasks.add_task(fire_webhook, MACRODROID_WEBHOOK_URL, action, "macrodroid")

    elif route == "deep":
        history = conversation_log.get(req.device_id, [])[-10:]
        context_block = "\n".join(f"[{h['role']}] {h['content']}" for h in history)
        deep_resp = await safe_generate(deep_model, f"{context_block}\n\nUser: {req.prompt}")
        result = {"source": "gemini_deep", "content": deep_resp.text}

    else:
        result = {"source": "gemini_direct", "content": routing.get("answer", "")}

    background_tasks.add_task(
        fire_webhook, DISCORD_WEBHOOK_URL,
        {"content": f"[{req.device_id}] {result['source']}: {req.prompt[:100]}"},
        "discord"
    )

    conversation_log.setdefault(req.device_id, []).append(
        {"role": "user", "content": req.prompt, "timestamp": time.time()}
    )
    conversation_log[req.device_id].append(
        {"role": "assistant", "content": result["content"], "timestamp": time.time()}
    )

    return result

@app.get("/api/health")
async def health():
    return {"status": "ok"}

# Serve frontend last so /api routes take priority
app.mount("/", StaticFiles(directory="static", html=True), name="static")
