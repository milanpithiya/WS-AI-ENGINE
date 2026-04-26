"""
Ollama LLM Client — Handles all communication with Ollama server.
Supports chat, model listing, health check, and streaming.
"""

from typing import Dict, List, Any, Optional
import asyncio
import json as _json
import time

import httpx
from loguru import logger


class OllamaClient:
    """Async HTTP client for Ollama API."""

    def __init__(self, config: Dict[str, Any]):
        ollama_cfg = config.get("ollama", {})
        self.base_url = ollama_cfg.get("base_url", "http://localhost:11434")
        self.default_model = ollama_cfg.get("default_model", "mistral")
        self.fallback_model = ollama_cfg.get("fallback_model", "llama3")
        self.timeout = ollama_cfg.get("timeout", 120)
        self.default_options = ollama_cfg.get("options", {})

        # Default timeout for non-chat operations (health check, list models).
        # Chat uses a per-request timeout — see chat().
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Health Check
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            response = await self._client.get("/api/tags")
            return response.status_code == 200
        except Exception as e:
            logger.warning("Ollama health check failed: {0}", str(e))
            return False

    # ------------------------------------------------------------------
    # List Models
    # ------------------------------------------------------------------

    async def list_models(self) -> List[Dict[str, Any]]:
        """List all models available in Ollama."""
        try:
            response = await self._client.get("/api/tags")
            if response.status_code == 200:
                data = response.json()
                return data.get("models", [])
            return []
        except Exception as e:
            logger.error("Failed to list models: {0}", str(e))
            return []

    # ------------------------------------------------------------------
    # Chat (Non-Streaming)
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Send a chat request to Ollama using streaming internally.

        Uses stream=True so that the read timeout resets on each token,
        preventing timeouts on slow CPU-only inference. Collects all
        chunks and returns the same structure as before.

        Returns:
            Dict with keys: answer, model_used, tokens_used, duration_ms
        """
        use_model = model or self.default_model
        use_options = dict(self.default_options)
        if options:
            use_options.update(options)

        payload = {
            "model": use_model,
            "messages": messages,
            "stream": True,
            "options": use_options,
        }

        logger.debug("Ollama chat | model={0} | messages={1} | options={2}",
                      use_model, len(messages), use_options)

        start_time = time.monotonic()

        # Streaming timeout strategy:
        # - read=None: Ollama buffers HTTP headers until after prompt eval
        #   completes (which can take minutes on CPU). An httpx read timeout
        #   here would kill the request during prompt evaluation.
        # - asyncio.timeout(self.timeout): caps the TOTAL wall-clock time
        #   (prompt eval + generation) so we don't hang indefinitely.
        # - Once streaming starts, each chunk arrives every ~300ms on CPU,
        #   well within any reasonable per-read timeout.
        stream_timeout = httpx.Timeout(connect=10.0, read=None, write=30.0, pool=30.0)

        try:
            async with asyncio.timeout(self.timeout):
                chunks: List[str] = []
                eval_count = 0
                prompt_eval_count = 0
                total_duration = 0

                async with self._client.stream("POST", "/api/chat", json=payload,
                                               timeout=stream_timeout) as response:
                    if response.status_code == 404 and use_model != self.fallback_model:
                        logger.warning("Model '{0}' not found, trying fallback '{1}'",
                                       use_model, self.fallback_model)
                        await response.aread()
                        return await self.chat(messages, model=self.fallback_model, options=options)

                    if response.status_code != 200:
                        error_text = (await response.aread()).decode(errors="replace")
                        logger.error("Ollama error | status={0} | body={1}",
                                     response.status_code, error_text[:500])
                        raise Exception("Ollama returned status {0}: {1}".format(
                            response.status_code, error_text[:200]))

                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = _json.loads(line)
                        except _json.JSONDecodeError:
                            continue

                        content = data.get("message", {}).get("content", "")
                        if content:
                            chunks.append(content)

                        if data.get("done", False):
                            eval_count = data.get("eval_count", 0)
                            prompt_eval_count = data.get("prompt_eval_count", 0)
                            total_duration = data.get("total_duration", 0)

            answer = "".join(chunks)
            duration_ms = int(total_duration / 1_000_000) if total_duration else int((time.monotonic() - start_time) * 1000)

            result = {
                "answer": answer,
                "model_used": use_model,
                "tokens_used": eval_count + prompt_eval_count,
                "prompt_tokens": prompt_eval_count,
                "completion_tokens": eval_count,
                "duration_ms": duration_ms,
            }

            logger.info("Ollama response | model={0} | tokens={1} | duration={2}ms",
                        use_model, result["tokens_used"], result["duration_ms"])
            return result

        except (httpx.TimeoutException, TimeoutError):
            elapsed = int(time.monotonic() - start_time)
            logger.error("Ollama request timed out after {0}s (timeout={1}s)", elapsed, self.timeout)
            raise Exception(
                "AI model took too long to respond. Try a shorter question or smaller dataset."
            )
        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama at {0}", self.base_url)
            raise Exception(
                "Cannot connect to AI model server at {0}. "
                "Please check if Ollama is running.".format(self.base_url)
            )

    # ------------------------------------------------------------------
    # Chat with Streaming (for future WebSocket integration)
    # ------------------------------------------------------------------

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        """
        Stream chat response from Ollama. Yields chunks as they arrive.

        Usage:
            async for chunk in client.chat_stream(messages):
                print(chunk)  # partial text
        """
        use_model = model or self.default_model
        use_options = dict(self.default_options)
        if options:
            use_options.update(options)

        payload = {
            "model": use_model,
            "messages": messages,
            "stream": True,
            "options": use_options,
        }

        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as response:
                import json
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            data = json.loads(line)
                            content = data.get("message", {}).get("content", "")
                            if content:
                                yield content
                            if data.get("done", False):
                                return
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error("Streaming error: {0}", str(e))
            raise

    # ------------------------------------------------------------------
    # Generate (single prompt, no chat format)
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Simple generate endpoint (non-chat format).
        Useful for single-shot completions.
        """
        use_model = model or self.default_model
        use_options = dict(self.default_options)
        if options:
            use_options.update(options)

        payload = {
            "model": use_model,
            "prompt": prompt,
            "stream": False,
            "options": use_options,
        }
        if system:
            payload["system"] = system

        try:
            response = await self._client.post("/api/generate", json=payload)
            if response.status_code == 200:
                data = response.json()
                return {
                    "answer": data.get("response", ""),
                    "model_used": use_model,
                    "tokens_used": data.get("eval_count", 0) + data.get("prompt_eval_count", 0),
                    "duration_ms": int(data.get("total_duration", 0) / 1_000_000),
                }
            raise Exception("Ollama generate error: {0}".format(response.text[:200]))
        except httpx.TimeoutException:
            raise Exception("AI model timed out. Try a shorter prompt.")
        except httpx.ConnectError:
            raise Exception("Cannot connect to Ollama at {0}".format(self.base_url))

    # ------------------------------------------------------------------
    # Pull Model (download a new model)
    # ------------------------------------------------------------------

    async def pull_model(self, model_name: str) -> bool:
        """Pull/download a model in Ollama. Returns True on success."""
        try:
            response = await self._client.post(
                "/api/pull",
                json={"name": model_name, "stream": False},
                timeout=httpx.Timeout(600.0),  # 10 min for large models
            )
            return response.status_code == 200
        except Exception as e:
            logger.error("Failed to pull model '{0}': {1}", model_name, str(e))
            return False
