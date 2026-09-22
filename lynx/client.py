"""Small Ollama chat client, using only the Python standard library."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LynxError(Exception):
    """A recoverable backend error."""


def chat(messages, model, host="http://localhost:11434", timeout=300):
    payload = {"model": model, "messages": messages, "stream": False}
    request = Request(
        host.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except HTTPError as exc:
        raise LynxError(
            f"Ollama returned HTTP {exc.code}. Check the server and installed "
            f"model ({model}); use `ollama list`."
        ) from exc
    except (URLError, OSError) as exc:
        raise LynxError(
            f"Cannot reach Ollama at {host}. Start it with `ollama serve`."
        ) from exc
    except (ValueError, UnicodeError) as exc:
        raise LynxError("Ollama returned invalid JSON.") from exc

    if not isinstance(result, dict):
        raise LynxError("Ollama returned an unexpected response.")
    if result.get("error"):
        raise LynxError(str(result["error"]))
    message = result.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise LynxError("Ollama response is missing message content.")
    return message["content"]
