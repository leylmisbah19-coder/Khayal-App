#!/usr/bin/env python3
"""
Khayal server -- runs locally OR as a deployed website (e.g. on Render.com).

Why a server exists at all: OpenAI's API doesn't send CORS headers, so a
browser can't call api.openai.com directly from a plain HTML page. This
server sits in between:

  1. Serves khayal.html (behind a password screen if APP_PASSWORD is set)
  2. POST /api/style          -> vision model reads the photo + measurements
     and invents 3 outfit concepts (no fixed catalog -- see note below)
  3. POST /api/generate-image -> image-EDIT model takes the user's own
     uploaded photo and changes only the clothing, keeping their face

Configuration is via environment variables (set these in Render's
dashboard, or `export` them locally before running):

  OPENAI_API_KEY  (required)  Your OpenAI key. Never sent to the browser.
  APP_PASSWORD    (optional)  If set, visitors must enter this password
                               before they can use the app -- important if
                               you're hosting this publicly, since every
                               request spends YOUR OpenAI credits.
  PORT            (optional)  Defaults to 8787 locally; hosting platforms
                               set this automatically.

Local run:
    export OPENAI_API_KEY=sk-...
    python3 khayal_proxy.py
    open http://localhost:8787

No pip installs needed -- everything here is Python's standard library,
which is also why it deploys cleanly to a bare Python web service.

Note on "real inventory": Dubai malls and most retailers don't expose a
public, live stock API, so this app can't truthfully say "this exact item
is in stock right now." Instead of faking that, the frontend links each
outfit concept out to a live search on real UAE retailers (Namshi, Zara
UAE, Ounass, H&M UAE, SHEIN UAE) so the user sees actual current listings.

Note on sessions: logged-in sessions are stored in memory. If the server
restarts (a free-tier host redeploying or waking from sleep does this),
everyone gets logged out and has to re-enter the password. That's fine
for a small assignment demo.
"""

import base64
import getpass
import http.cookies
import json
import mimetypes
import os
import secrets
import urllib.error
import urllib.request
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PORT", 8787))
HOST = os.environ.get("HOST", "0.0.0.0")
HTML_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "khayal.html")

# If either model name 404s on your account, swap it for whatever
# vision-capable chat model / image-editing model you have access to.
CHAT_MODEL = "gpt-5-mini"
IMAGE_MODEL = "gpt-image-1"

API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    # Only prompt interactively if this looks like a local terminal run.
    try:
        API_KEY = getpass.getpass(
            "Enter your OpenAI API key (hidden input, kept in memory for this run only): "
        ).strip()
    except Exception:
        pass
if not API_KEY:
    raise SystemExit(
        "No OpenAI API key found. Set the OPENAI_API_KEY environment variable "
        "(in Render's dashboard if hosted, or via `export OPENAI_API_KEY=sk-...` locally)."
    )

APP_PASSWORD = os.environ.get("APP_PASSWORD")  # if unset, app runs open (fine for local-only use)
SESSION_COOKIE = "khayal_session"
VALID_SESSIONS = set()  # in-memory; cleared on restart, see note above

LOGIN_PAGE = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Khayal — Sign in</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body{{font-family:-apple-system,sans-serif;background:#f4ecdd;display:flex;
       align-items:center;justify-content:center;min-height:100vh;margin:0;}}
  form{{background:#fffdf8;border:1px solid #e4d7b8;border-radius:14px;padding:32px;
        width:280px;text-align:center;}}
  h1{{margin:0 0 4px;font-size:1.4rem;}}
  p{{color:#6b6250;font-size:.85rem;margin:0 0 18px;}}
  input{{width:100%;padding:10px;border:1px solid #e4d7b8;border-radius:8px;
        margin-bottom:12px;box-sizing:border-box;font-size:.9rem;}}
  button{{width:100%;padding:10px;background:#1c1a17;color:#fff;border:none;
         border-radius:8px;font-weight:600;cursor:pointer;}}
  .err{{color:#a1341f;font-size:.8rem;margin-bottom:10px;}}
</style></head>
<body>
  <form method="POST" action="/api/login">
    <h1>Khayal <span style="color:#b7913a;">خيال</span></h1>
    <p>Enter the app password to continue.</p>
    {error_html}
    <input type="password" name="password" placeholder="Password" autofocus required>
    <button type="submit">Enter</button>
  </form>
</body></html>
"""

SYSTEM_PROMPT = """You are a styling assistant for a fashion app aimed at a Dubai
audience. You'll be given a person's age, height, weight, a style preference,
whether they want modest-only outfits, and a photo. Your job:

1. In one or two SHORT, neutral, practical sentences, note body proportions
   relevant to how clothes will fit (e.g. torso length, shoulder width, frame).
   NEVER comment on attractiveness, weight, or suggest the person change their
   body -- describe proportions the way a tailor would, only to justify fit choices.
2. Invent exactly 3 distinct outfit concepts appropriate for their age and style
   preference, drawing on current fashion trends relevant to Dubai's market:
   modest fashion staples like kimono-cut and floral-panel abayas, belted
   minimal abayas, and utility maxi dresses; alongside global trends like
   pleated co-ord sets, wide-leg denim, oversized streetwear layering, satin
   slip dresses, tailored blazer dresses, and y2k-influenced metallics. Use
   these as inspiration, not a fixed list -- invent outfits that make sense
   for THIS person. If modest-only was requested, every concept must be modest
   (long sleeves/hem, non-sheer, loose through the body).

For each outfit give:
  - "name": short outfit name (e.g. "Belted Camel Abaya")
  - "category": one or two words (e.g. "abaya", "streetwear", "evening dress")
  - "description": one vivid sentence describing the exact garment(s), fabric,
    color, and cut -- specific enough to use as an image-generation prompt
  - "searchKeywords": 2-4 words a person would type into a fashion retailer's
    search bar to find something like this (e.g. "belted camel abaya")
  - "reason": one sentence on why this suits their age, proportions, and style
    preference

Respond ONLY with strict JSON, no markdown fences, in this exact shape:
{"bodyNotes": "string", "picks": [
  {"name": "string", "category": "string", "description": "string",
   "searchKeywords": "string", "reason": "string"}, ... exactly 3 ...
]}
"""


def call_openai_chat(payload):
    photo_b64 = payload["photoBase64"]
    photo_mime = payload.get("photoMime") or "image/jpeg"
    modest_line = "Modest-only: yes, every concept must be modest." if payload.get("modestOnly") else "Modest-only: no."
    user_text = (
        f"Age: {payload.get('age')}\n"
        f"Height: {payload.get('height')} cm\n"
        f"Weight: {payload.get('weight')} kg\n"
        f"Style preference: {payload.get('stylePref')}\n"
        f"{modest_line}"
    )

    body = {
        "model": CHAT_MODEL,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:{photo_mime};base64,{photo_b64}"}},
                ],
            },
        ],
    }

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    raw = result["choices"][0]["message"]["content"].strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


def build_multipart(fields, file_field_name, filename, file_bytes, file_mime):
    """Minimal multipart/form-data encoder (stdlib only)."""
    boundary = uuid.uuid4().hex
    parts = []
    for key, value in fields.items():
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n'
            f"{value}\r\n".encode("utf-8")
        )
    parts.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{file_field_name}"; filename="{filename}"\r\n'
            f"Content-Type: {file_mime}\r\n\r\n"
        ).encode("utf-8")
        + file_bytes
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(parts)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def call_openai_image_edit(prompt, photo_b64, photo_mime):
    file_bytes = base64.b64decode(photo_b64)
    ext = mimetypes.guess_extension(photo_mime) or ".jpg"
    filename = f"photo{ext}"

    fields = {
        "model": IMAGE_MODEL,
        "prompt": prompt,
        "size": "1024x1536",
        "quality": "medium",
        "n": "1",
    }
    body, content_type = build_multipart(fields, "image", filename, file_bytes, photo_mime)

    req = urllib.request.Request(
        "https://api.openai.com/v1/images/edits",
        data=body,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": content_type},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result["data"][0]["b64_json"]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("  " + (fmt % args))

    # ---------- auth helpers ----------
    def _is_authed(self):
        if not APP_PASSWORD:
            return True  # no password configured -> app runs open
        raw_cookie = self.headers.get("Cookie")
        if not raw_cookie:
            return False
        jar = http.cookies.SimpleCookie()
        jar.load(raw_cookie)
        morsel = jar.get(SESSION_COOKIE)
        return bool(morsel and morsel.value in VALID_SESSIONS)

    def _serve_login(self, error=False):
        error_html = '<p class="err">Wrong password — try again.</p>' if error else ""
        content = LOGIN_PAGE.format(error_html=error_html).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _handle_login(self, raw_body):
        fields = urllib.parse.parse_qs(raw_body.decode("utf-8"))
        submitted = (fields.get("password") or [""])[0]
        if APP_PASSWORD and secrets.compare_digest(submitted, APP_PASSWORD):
            token = secrets.token_hex(24)
            VALID_SESSIONS.add(token)
            self.send_response(303)
            self.send_header("Location", "/")
            self.send_header(
                "Set-Cookie",
                f"{SESSION_COOKIE}={token}; HttpOnly; Path=/; Max-Age=86400; SameSite=Lax",
            )
            self.end_headers()
        else:
            self._serve_login(error=True)

    # ---------- json helper ----------
    def _send_json(self, status, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/khayal.html"):
            if not self._is_authed():
                self._serve_login()
                return
            try:
                with open(HTML_FILE, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            except FileNotFoundError:
                self._send_json(404, {"error": "khayal.html not found next to this script"})
        elif self.path == "/logout":
            self.send_response(303)
            self.send_header("Location", "/")
            self.send_header("Set-Cookie", f"{SESSION_COOKIE}=; Path=/; Max-Age=0")
            self.end_headers()
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)

        if self.path == "/api/login":
            self._handle_login(raw)
            return

        if not self._is_authed():
            self._send_json(401, {"error": "Not signed in. Go to / and enter the password first."})
            return

        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self._send_json(400, {"error": "bad JSON body"})
            return

        try:
            if self.path == "/api/style":
                result = call_openai_chat(payload)
                self._send_json(200, result)
            elif self.path == "/api/generate-image":
                b64 = call_openai_image_edit(payload["prompt"], payload["photoBase64"], payload.get("photoMime") or "image/jpeg")
                self._send_json(200, {"image": b64})
            else:
                self._send_json(404, {"error": "not found"})
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            print("OpenAI API error:", detail)
            self._send_json(e.code, {"error": f"OpenAI API error {e.code}", "detail": detail})
        except Exception as e:
            print("Server error:", e)
            self._send_json(500, {"error": str(e)})


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"\nKhayal running at http://{HOST}:{PORT}  (Ctrl+C to stop)")
    if APP_PASSWORD:
        print("Password protection is ON.")
    else:
        print("No APP_PASSWORD set -- app is open to anyone who reaches this URL.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
