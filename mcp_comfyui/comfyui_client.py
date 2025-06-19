"""ComfyUI API client implementation."""

import json
import uuid
import asyncio
import logging
from typing import Any, Dict, List, Optional

import aiohttp
import websockets

logger = logging.getLogger(__name__)


class ComfyUIClient:
    """Client for interacting with ComfyUI API."""

    def __init__(self, host: str = "localhost", port: int = 8188):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}/ws"
        self.client_id = str(uuid.uuid4())

    async def queue_prompt(self, workflow: Dict[str, Any]) -> str:
        """Queue a workflow for execution."""
        prompt_id = str(uuid.uuid4())
        payload = {
            "prompt": workflow,
            "client_id": self.client_id,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/prompt",
                json=payload,
            ) as response:
                result = await response.json()
                return result.get("prompt_id", prompt_id)

    async def get_queue(self) -> Dict[str, Any]:
        """Get current queue status."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/queue") as response:
                return await response.json()

    async def get_history(self, limit: int = 10, prompt_id: Optional[str] = None) -> Dict[str, Any]:
        """Get generation history."""
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/history"
            if prompt_id:
                url += f"/{prompt_id}"
            async with session.get(url) as response:
                history = await response.json()
                
                # Limit results if no specific prompt_id
                if not prompt_id and isinstance(history, dict):
                    items = list(history.items())[:limit]
                    history = dict(items)
                
                return history

    async def get_models(self) -> List[str]:
        """Get list of available model checkpoints."""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/object_info") as response:
                data = await response.json()
                
                # Extract checkpoint models
                checkpoint_loader = data.get("CheckpointLoaderSimple", {})
                input_info = checkpoint_loader.get("input", {}).get("required", {})
                ckpt_info = input_info.get("ckpt_name", [[]])
                
                if ckpt_info and len(ckpt_info) > 0:
                    return ckpt_info[0]
                
                return []

    async def interrupt(self) -> bool:
        """Interrupt current generation."""
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.base_url}/interrupt") as response:
                return response.status == 200

    async def wait_for_completion(self, prompt_id: str, timeout: int = 300) -> Dict[str, Any]:
        """Wait for a prompt to complete using WebSocket."""
        try:
            async with websockets.connect(f"{self.ws_url}?clientId={self.client_id}") as websocket:
                start_time = asyncio.get_event_loop().time()
                
                while True:
                    if asyncio.get_event_loop().time() - start_time > timeout:
                        raise TimeoutError(f"Generation timeout after {timeout} seconds")
                    
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)
                        
                        if data.get("type") == "executed" and data.get("data", {}).get("prompt_id") == prompt_id:
                            # Get the final result from history
                            history = await self.get_history(prompt_id=prompt_id)
                            return history.get(prompt_id, {})
                            
                    except asyncio.TimeoutError:
                        # Check if still in queue
                        queue = await self.get_queue()
                        if prompt_id not in str(queue):
                            # Not in queue, check history
                            history = await self.get_history(prompt_id=prompt_id)
                            if prompt_id in history:
                                return history[prompt_id]
                        continue
                        
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            # Fallback to polling
            return await self._poll_for_completion(prompt_id, timeout)

    async def _poll_for_completion(self, prompt_id: str, timeout: int = 300) -> Dict[str, Any]:
        """Fallback polling method for completion."""
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            history = await self.get_history(prompt_id=prompt_id)
            if prompt_id in history:
                return history[prompt_id]
            
            await asyncio.sleep(1)
        
        raise TimeoutError(f"Generation timeout after {timeout} seconds")
