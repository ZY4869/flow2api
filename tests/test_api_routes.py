import asyncio
import base64
import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.api import admin as admin_module
from src.api import routes
from src.core.auth import AuthManager, verify_api_key_flexible


def build_openai_completion(content: str) -> str:
    return json.dumps(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "flow2api",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
        }
    )


def test_openai_route_resolves_alias_and_returns_non_stream_result(client, fake_handler):
    fake_handler.non_stream_chunks = [build_openai_completion("![Generated Image](https://example.com/out.png)")]

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemini-3.0-pro-image",
            "messages": [{"role": "user", "content": "draw a sunset"}],
            "generationConfig": {
                "imageConfig": {
                    "aspectRatio": "16:9",
                    "imageSize": "2K",
                }
            },
        },
    )

    assert response.status_code == 200
    assert fake_handler.calls[0]["model"] == "gemini-3.0-pro-image-landscape-2k"
    assert response.json()["choices"][0]["message"]["content"].startswith("![Generated Image]")


def test_openai_route_returns_handler_error_status(client, fake_handler):
    fake_handler.non_stream_chunks = [
        json.dumps(
            {
                "error": {
                    "message": "没有可用的Token进行图片生成",
                    "status_code": 503,
                }
            }
        )
    ]

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemini-3.0-pro-image",
            "messages": [{"role": "user", "content": "draw a tree"}],
        },
    )

    assert response.status_code == 503
    assert response.json()["error"]["message"] == "没有可用的Token进行图片生成"


def test_flexible_auth_accepts_x_goog_api_key(monkeypatch):
    monkeypatch.setattr(AuthManager, "verify_api_key", staticmethod(lambda api_key: api_key == "secret"))

    assert asyncio.run(
        verify_api_key_flexible(
            credentials=None,
            x_goog_api_key="secret",
            key=None,
        )
    ) == "secret"


def test_admin_remote_browser_helper_uses_asyncsession(monkeypatch):
    calls = []

    class FakeResponse:
        status_code = 200
        text = '{"success": true, "token": "abc"}'

        def json(self):
            return {"success": True, "token": "abc"}

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def request(self, method, url, **kwargs):
            calls.append({
                "method": method,
                "url": url,
                "kwargs": kwargs,
            })
            return FakeResponse()

    monkeypatch.setattr(admin_module, "AsyncSession", FakeSession)

    status_code, payload, response_text = asyncio.run(
        admin_module._sync_json_http_request(
            method="POST",
            url="https://example.com/api/v1/custom-score",
            headers={"Authorization": "Bearer token"},
            payload={"website_url": "https://example.com"},
            timeout=15,
        )
    )

    assert status_code == 200
    assert payload == {"success": True, "token": "abc"}
    assert response_text == '{"success": true, "token": "abc"}'
    assert calls == [
        {
            "method": "POST",
            "url": "https://example.com/api/v1/custom-score",
            "kwargs": {
                "headers": {
                    "Authorization": "Bearer token",
                    "Accept": "application/json",
                    "Content-Type": "application/json; charset=utf-8",
                },
                "timeout": 15,
                "impersonate": "chrome120",
                "json": {"website_url": "https://example.com"},
            },
        }
    ]


class FakeCaptchaConfigDB:
    def __init__(self, captcha_method: str = "browser"):
        self.captcha_method = captcha_method

    async def get_captcha_config(self):
        return SimpleNamespace(
            captcha_method=self.captcha_method,
            yescaptcha_api_key="",
            yescaptcha_base_url="https://api.yescaptcha.com",
            capmonster_api_key="",
            capmonster_base_url="https://api.capmonster.cloud",
            ezcaptcha_api_key="",
            ezcaptcha_base_url="https://api.ez-captcha.com",
            capsolver_api_key="",
            capsolver_base_url="https://api.capsolver.com",
            remote_browser_base_url="",
            remote_browser_api_key="",
            remote_browser_timeout=60,
            browser_proxy_enabled=False,
            browser_proxy_url="",
            browser_count=1,
        )


def build_admin_client():
    app = FastAPI()
    app.include_router(admin_module.router)

    async def fake_admin_auth():
        return "test-admin-token"

    app.dependency_overrides[admin_module.verify_admin_token] = fake_admin_auth
    return TestClient(app)


def test_admin_captcha_config_exposes_blocked_browser_runtime(monkeypatch):
    monkeypatch.setattr(admin_module, "db", FakeCaptchaConfigDB("browser"))
    monkeypatch.setattr(admin_module, "_is_running_in_docker", lambda: True)
    monkeypatch.delenv("ALLOW_DOCKER_HEADED_CAPTCHA", raising=False)
    monkeypatch.delenv("ALLOW_DOCKER_BROWSER_CAPTCHA", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)

    with build_admin_client() as client:
        response = client.get("/api/captcha/config")

    assert response.status_code == 200
    payload = response.json()
    runtime = payload["captcha_runtime"]["browser"]
    assert runtime["available"] is False
    assert runtime["docker_headed_blocked"] is True
    assert "docker-compose.headed.yml" in runtime["message"]


def test_admin_score_test_short_circuits_when_browser_runtime_blocked(monkeypatch):
    monkeypatch.setattr(admin_module, "db", FakeCaptchaConfigDB("browser"))
    monkeypatch.setattr(admin_module, "_is_running_in_docker", lambda: True)
    monkeypatch.delenv("ALLOW_DOCKER_HEADED_CAPTCHA", raising=False)
    monkeypatch.delenv("ALLOW_DOCKER_BROWSER_CAPTCHA", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)

    with build_admin_client() as client:
        response = client.post("/api/captcha/score-test", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is False
    assert payload["token_acquired"] is False
    assert payload["runtime"]["available"] is False
    assert "docker-compose.headed.yml" in payload["message"]


def test_openai_route_returns_base64_image_when_requested(client, fake_handler, monkeypatch):
    fake_handler.non_stream_chunks = [
        build_openai_completion("![Generated Image](https://example.com/out.png)")
    ]

    async def fake_retrieve_image_data(url: str):
        return b"\x89PNG\r\n\x1a\nopenai"

    monkeypatch.setattr(routes, "retrieve_image_data", fake_retrieve_image_data)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gemini-3.0-pro-image",
            "messages": [{"role": "user", "content": "draw a sunset"}],
            "responseImageEncoding": "base64",
        },
    )

    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert content.startswith("![Generated Image](data:image/png;base64,")
    encoded = content.split("base64,", 1)[1].rstrip(")")
    assert base64.b64decode(encoded).startswith(b"\x89PNG")
    assert response.headers["X-Flow2API-Image-Encoding-Applied"] == "base64"


def test_openai_route_body_image_encoding_overrides_header(client, fake_handler):
    fake_handler.non_stream_chunks = [
        build_openai_completion("![Generated Image](https://example.com/out.png)")
    ]

    response = client.post(
        "/v1/chat/completions",
        headers={"X-Flow2API-Image-Encoding": "base64"},
        json={
            "model": "gemini-3.0-pro-image",
            "messages": [{"role": "user", "content": "draw a sunset"}],
            "responseImageEncoding": "url",
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "![Generated Image](https://example.com/out.png)"
    assert response.headers["X-Flow2API-Image-Encoding-Applied"] == "url"


def test_openai_stream_only_rewrites_final_image_chunk(client, fake_handler, monkeypatch):
    fake_handler.stream_chunks = [
        'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,"model":"flow2api","choices":[{"index":0,"delta":{"reasoning_content":"starting generation"},"finish_reason":null}]}\n\n',
        'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,"model":"flow2api","choices":[{"index":0,"delta":{"content":"![Generated Image](https://example.com/final.png)"},"finish_reason":"stop"}]}\n\n',
    ]

    async def fake_retrieve_image_data(url: str):
        return b"\x89PNG\r\n\x1a\nstream"

    monkeypatch.setattr(routes, "retrieve_image_data", fake_retrieve_image_data)

    response = client.post(
        "/v1/chat/completions",
        headers={"X-Flow2API-Image-Encoding": "base64"},
        json={
            "model": "gemini-3.0-pro-image",
            "messages": [{"role": "user", "content": "draw a city"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert response.headers["X-Flow2API-Image-Encoding-Applied"] == "base64"
    data_lines = [
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]

    first_chunk = json.loads(data_lines[0])
    assert first_chunk["choices"][0]["delta"]["reasoning_content"] == "starting generation"

    second_chunk = json.loads(data_lines[1])
    assert second_chunk["choices"][0]["delta"]["content"].startswith(
        "![Generated Image](data:image/png;base64,"
    )


def test_openai_video_response_stays_as_video_markup(client, fake_handler):
    fake_handler.non_stream_chunks = [
        build_openai_completion(
            "```html\n<video src='https://example.com/video.mp4' controls></video>\n```"
        )
    ]

    response = client.post(
        "/v1/chat/completions",
        headers={"X-Flow2API-Image-Encoding": "base64"},
        json={
            "model": "veo_3_1_t2v_fast_landscape",
            "messages": [{"role": "user", "content": "make a video"}],
        },
    )

    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert "https://example.com/video.mp4" in content
    assert "data:video" not in content


class FakeResponseConfigDB:
    def __init__(self, image_encoding: str = "url"):
        self.image_encoding = image_encoding
        self.reload_calls = 0

    async def get_response_config(self):
        return SimpleNamespace(image_encoding=self.image_encoding)

    async def update_response_config(self, image_encoding: str):
        self.image_encoding = image_encoding

    async def reload_config_to_memory(self):
        self.reload_calls += 1


def test_admin_response_config_endpoints_round_trip(monkeypatch):
    fake_db = FakeResponseConfigDB("url")
    monkeypatch.setattr(admin_module, "db", fake_db)

    with build_admin_client() as client:
        get_response = client.get("/api/response/config")
        post_response = client.post(
            "/api/response/config",
            json={"image_encoding": "base64"},
        )

    assert get_response.status_code == 200
    assert get_response.json()["config"]["image_encoding"] == "url"
    assert post_response.status_code == 200
    assert post_response.json()["config"]["image_encoding"] == "base64"
    assert fake_db.image_encoding == "base64"
    assert fake_db.reload_calls == 1
