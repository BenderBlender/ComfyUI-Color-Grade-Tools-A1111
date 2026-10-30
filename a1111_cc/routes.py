"""HTTP routes the grading panel uses to keep presets.

Registered on ComfyUI's own server when it is running, and quietly skipped
when this package is imported anywhere else — tests included.
"""

import logging

from . import presets

log = logging.getLogger(__name__)


def register() -> bool:
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return False

    instance = getattr(PromptServer, "instance", None)
    if instance is None or not hasattr(instance, "routes"):
        return False

    routes = instance.routes

    @routes.get("/colorkit/presets")
    async def list_presets(request):            # noqa: ARG001
        return web.json_response({
            "builtin": {f"· {k}": v for k, v in presets.BUILT_IN.items()},
            "user": presets.load_user(),
        })

    @routes.post("/colorkit/presets")
    async def save_preset(request):
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001
            return web.json_response({"error": "expected JSON"}, status=400)

        name = presets.clean_name(payload.get("name", ""))
        if not name:
            return web.json_response({"error": "a preset needs a name"}, status=400)

        values = payload.get("values")
        if not isinstance(values, dict):
            return web.json_response({"error": "values must be an object"}, status=400)

        stored = presets.load_user()
        stored[name] = presets.sanitise(values)
        try:
            presets.save_user(stored)
        except OSError as exc:
            log.exception("Color Grade: could not save presets")
            return web.json_response({"error": str(exc)}, status=500)
        return web.json_response({"user": stored})

    @routes.delete("/colorkit/presets")
    async def delete_preset(request):
        name = presets.clean_name(request.query.get("name", ""))
        stored = presets.load_user()
        if name not in stored:
            return web.json_response({"error": "no such preset"}, status=404)
        del stored[name]
        try:
            presets.save_user(stored)
        except OSError as exc:
            return web.json_response({"error": str(exc)}, status=500)
        return web.json_response({"user": stored})

    log.info("Color Grade: preset routes ready")
    return True
