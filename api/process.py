from http.server import BaseHTTPRequestHandler
import json
import os
import urllib.request
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

router_model = genai.GenerativeModel(
    "gemini-flash-latest",
    generation_config={"response_mime_type": "application/json"}
)
deep_model = genai.GenerativeModel(
    "gemini-pro-latest",
    tools="google_search_retrieval"
)

ROUTER_PROMPT = """You are a routing controller for a personal AI assistant. Output ONLY valid JSON, no markdown, no code fences, no prose:
{"route": "local"|"direct"|"deep", "webhook_action": string|null, "answer": string|null}

route=local -> prompt maps to a known device automation command (e.g. "turn on lights", "toggle wifi")
route=direct -> short factual/conversational answer, put it in "answer". Be accurate and concise.
route=deep -> needs multi-step reasoning, code, long-form output, or current/real-time information (news, prices, current office holders, recent events)

Important: your training data has a cutoff and may be outdated on time-sensitive topics. If a direct question involves anything that could have changed recently, route it to "deep" instead of guessing.
"""

def fire_webhook(url, payload):
    if not url:
        return
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        print(f"webhook failed: {e}")

def safe_generate(model, prompt, retries=2):
    last_err = None
    for i in range(retries):
        try:
            return model.generate_content(prompt)
        except Exception as e:
            last_err = e
    raise last_err

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            prompt = body.get("prompt", "")
            device_id = body.get("device_id", "unknown")
            device_type = body.get("device_type", "unknown")

            if not prompt:
                self._send(400, {"error": "prompt is required"})
                return

            full_prompt = f"{ROUTER_PROMPT}\nDevice: {device_type}\nPrompt: {prompt}"
            resp = safe_generate(router_model, full_prompt)

            try:
                routing = json.loads(resp.text)
            except json.JSONDecodeError:
                routing = {"route": "direct", "answer": resp.text}

            route = routing.get("route", "direct")

            if route == "local":
                action = routing.get("webhook_action")
                result = {"source": "local", "content": action or "No action specified"}
                fire_webhook(os.environ.get("MACRODROID_WEBHOOK_URL"), {"action": action})

            elif route == "deep":
                deep_resp = safe_generate(deep_model, prompt)
                result = {"source": "gemini_deep", "content": deep_resp.text}

            else:
                result = {"source": "gemini_direct", "content": routing.get("answer", "")}

            fire_webhook(
                os.environ.get("DISCORD_WEBHOOK_URL"),
                {"content": f"[{device_id}] {result['source']}: {prompt[:100]}"}
            )

            self._send(200, result)

        except Exception as e:
            self._send(500, {"error": str(e)})

    def _send(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
