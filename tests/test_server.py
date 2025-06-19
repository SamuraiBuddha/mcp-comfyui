"""Tests for MCP ComfyUI server."""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch

from mcp_comfyui.server import ComfyUIServer
from mcp_comfyui.comfyui_client import ComfyUIClient


@pytest.fixture
def server():
    """Create a test server instance."""
    return ComfyUIServer()


@pytest.fixture
def mock_client():
    """Create a mock ComfyUI client."""
    client = Mock(spec=ComfyUIClient)
    client.queue_prompt = AsyncMock(return_value="test-prompt-id")
    client.wait_for_completion = AsyncMock(return_value={"status": "completed"})
    client.get_queue = AsyncMock(return_value={"queue_remaining": 0})
    client.get_history = AsyncMock(return_value={})
    client.get_models = AsyncMock(return_value=["sd_xl_base_1.0.safetensors"])
    client.interrupt = AsyncMock(return_value=True)
    return client


@pytest.mark.asyncio
async def test_generate_image(server, mock_client):
    """Test image generation."""
    server.client = mock_client
    
    result = await server._generate_image({
        "prompt": "test prompt",
        "width": 512,
        "height": 512,
    })
    
    assert result["prompt_id"] == "test-prompt-id"
    assert result["status"] == "completed"
    mock_client.queue_prompt.assert_called_once()
    mock_client.wait_for_completion.assert_called_once_with("test-prompt-id")


@pytest.mark.asyncio
async def test_list_workflows(server):
    """Test workflow listing."""
    with patch("pathlib.Path.exists", return_value=True):
        with patch("pathlib.Path.glob", return_value=[
            Mock(name="workflow1.json"),
            Mock(name="workflow2.json"),
        ]):
            workflows = await server._list_workflows()
            assert len(workflows) == 2
            assert "workflow1.json" in workflows
            assert "workflow2.json" in workflows


def test_build_default_workflow(server):
    """Test default workflow generation."""
    workflow = server._build_default_workflow(
        prompt="test prompt",
        negative_prompt="bad quality",
        width=1024,
        height=1024,
        steps=30,
        cfg_scale=8.0,
        seed=12345,
    )
    
    assert "1" in workflow  # CheckpointLoader
    assert "2" in workflow  # Positive prompt
    assert "3" in workflow  # Negative prompt
    assert "5" in workflow  # KSampler
    
    # Check values
    assert workflow["2"]["inputs"]["text"] == "test prompt"
    assert workflow["3"]["inputs"]["text"] == "bad quality"
    assert workflow["5"]["inputs"]["seed"] == 12345
    assert workflow["5"]["inputs"]["steps"] == 30
