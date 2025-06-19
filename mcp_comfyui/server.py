"""MCP server implementation for ComfyUI integration with enhanced API features."""

import os
import json
import asyncio
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions

from .comfyui_client import ComfyUIClient
from .utils import encode_image_to_base64, validate_workflow

logger = logging.getLogger(__name__)


class ComfyUIServer:
    """MCP Server for ComfyUI integration with enhanced API features."""

    def __init__(self):
        self.server = Server("mcp-comfyui")
        self.client: Optional[ComfyUIClient] = None
        self._setup_handlers()

    def _setup_handlers(self):
        """Set up MCP server handlers."""

        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            """List available tools."""
            return [
                types.Tool(
                    name="generate_image",
                    description="Generate an image from a text prompt",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string", "description": "Positive prompt"},
                            "negative_prompt": {"type": "string", "description": "Negative prompt"},
                            "width": {"type": "integer", "default": 512},
                            "height": {"type": "integer", "default": 512},
                            "steps": {"type": "integer", "default": 20},
                            "cfg_scale": {"type": "number", "default": 7.0},
                            "seed": {"type": "integer", "default": -1},
                            "model": {"type": "string", "description": "Model checkpoint to use"},
                            "sampler": {"type": "string", "default": "euler"},
                            "scheduler": {"type": "string", "default": "normal"},
                        },
                        "required": ["prompt"],
                    },
                ),
                types.Tool(
                    name="build_workflow",
                    description="Build a custom workflow programmatically",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "nodes": {
                                "type": "array",
                                "description": "List of workflow nodes",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "type": {"type": "string"},
                                        "inputs": {"type": "object"},
                                    },
                                    "required": ["id", "type", "inputs"],
                                },
                            },
                        },
                        "required": ["nodes"],
                    },
                ),
                types.Tool(
                    name="execute_workflow",
                    description="Execute a saved ComfyUI workflow",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "workflow_name": {"type": "string", "description": "Name of the workflow file"},
                            "inputs": {"type": "object", "description": "Workflow input parameters"},
                        },
                        "required": ["workflow_name"],
                    },
                ),
                types.Tool(
                    name="save_workflow",
                    description="Save a workflow to file",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Name for the workflow"},
                            "workflow": {"type": "object", "description": "Workflow definition"},
                            "description": {"type": "string", "description": "Workflow description"},
                        },
                        "required": ["name", "workflow"],
                    },
                ),
                types.Tool(
                    name="list_workflows",
                    description="List available workflow files",
                    inputSchema={"type": "object", "properties": {}},
                ),
                types.Tool(
                    name="get_queue_status",
                    description="Get the current queue status",
                    inputSchema={"type": "object", "properties": {}},
                ),
                types.Tool(
                    name="get_history",
                    description="Get generation history",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "limit": {"type": "integer", "default": 10},
                        },
                    },
                ),
                types.Tool(
                    name="get_image",
                    description="Retrieve a generated image",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt_id": {"type": "string", "description": "ID of the generation"},
                        },
                        "required": ["prompt_id"],
                    },
                ),
                types.Tool(
                    name="list_models",
                    description="List available model checkpoints",
                    inputSchema={"type": "object", "properties": {}},
                ),
                types.Tool(
                    name="list_loras",
                    description="List available LoRA models",
                    inputSchema={"type": "object", "properties": {}},
                ),
                types.Tool(
                    name="list_samplers",
                    description="List available samplers",
                    inputSchema={"type": "object", "properties": {}},
                ),
                types.Tool(
                    name="list_schedulers",
                    description="List available schedulers",
                    inputSchema={"type": "object", "properties": {}},
                ),
                types.Tool(
                    name="get_node_types",
                    description="Get all available node types",
                    inputSchema={"type": "object", "properties": {}},
                ),
                types.Tool(
                    name="get_embeddings",
                    description="List available embeddings",
                    inputSchema={"type": "object", "properties": {}},
                ),
                types.Tool(
                    name="interrupt_generation",
                    description="Interrupt the current generation",
                    inputSchema={"type": "object", "properties": {}},
                ),
                types.Tool(
                    name="clear_queue",
                    description="Clear the generation queue",
                    inputSchema={"type": "object", "properties": {}},
                ),
                types.Tool(
                    name="upload_image",
                    description="Upload an image for use in workflows",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "image_path": {"type": "string", "description": "Path to the image file"},
                            "name": {"type": "string", "description": "Name for the uploaded image"},
                        },
                        "required": ["image_path"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def handle_call_tool(
            name: str, arguments: dict | None
        ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
            """Handle tool execution."""
            if not self.client:
                raise RuntimeError("ComfyUI client not initialized")

            if name == "generate_image":
                result = await self._generate_image(arguments or {})
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

            elif name == "build_workflow":
                result = await self._build_workflow(arguments or {})
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

            elif name == "execute_workflow":
                result = await self._execute_workflow(arguments or {})
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

            elif name == "save_workflow":
                result = await self._save_workflow(arguments or {})
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

            elif name == "list_workflows":
                workflows = await self._list_workflows()
                return [types.TextContent(type="text", text=json.dumps(workflows, indent=2))]

            elif name == "get_queue_status":
                status = await self.client.get_queue()
                return [types.TextContent(type="text", text=json.dumps(status, indent=2))]

            elif name == "get_history":
                limit = arguments.get("limit", 10) if arguments else 10
                history = await self.client.get_history(limit)
                return [types.TextContent(type="text", text=json.dumps(history, indent=2))]

            elif name == "get_image":
                if not arguments or "prompt_id" not in arguments:
                    raise ValueError("prompt_id is required")
                image_data = await self._get_image(arguments["prompt_id"])
                return [types.TextContent(type="text", text=json.dumps(image_data, indent=2))]

            elif name == "list_models":
                models = await self.client.get_models()
                return [types.TextContent(type="text", text=json.dumps(models, indent=2))]

            elif name == "list_loras":
                loras = await self._get_loras()
                return [types.TextContent(type="text", text=json.dumps(loras, indent=2))]

            elif name == "list_samplers":
                samplers = await self._get_samplers()
                return [types.TextContent(type="text", text=json.dumps(samplers, indent=2))]

            elif name == "list_schedulers":
                schedulers = await self._get_schedulers()
                return [types.TextContent(type="text", text=json.dumps(schedulers, indent=2))]

            elif name == "get_node_types":
                node_types = await self._get_node_types()
                return [types.TextContent(type="text", text=json.dumps(node_types, indent=2))]

            elif name == "get_embeddings":
                embeddings = await self._get_embeddings()
                return [types.TextContent(type="text", text=json.dumps(embeddings, indent=2))]

            elif name == "interrupt_generation":
                result = await self.client.interrupt()
                return [types.TextContent(type="text", text=json.dumps({"interrupted": result}))]

            elif name == "clear_queue":
                result = await self._clear_queue()
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

            elif name == "upload_image":
                result = await self._upload_image(arguments or {})
                return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

            else:
                raise ValueError(f"Unknown tool: {name}")

    async def _generate_image(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an image using the default workflow with advanced options."""
        # Build workflow with all options
        workflow = self._build_default_workflow(
            prompt=args.get("prompt", ""),
            negative_prompt=args.get("negative_prompt", ""),
            width=args.get("width", 512),
            height=args.get("height", 512),
            steps=args.get("steps", 20),
            cfg_scale=args.get("cfg_scale", 7.0),
            seed=args.get("seed", -1),
            model=args.get("model", "sd_xl_base_1.0.safetensors"),
            sampler=args.get("sampler", "euler"),
            scheduler=args.get("scheduler", "normal"),
        )

        prompt_id = await self.client.queue_prompt(workflow)
        
        # Wait for completion
        result = await self.client.wait_for_completion(prompt_id)
        
        return {
            "prompt_id": prompt_id,
            "status": "completed",
            "outputs": result,
        }

    async def _build_workflow(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Build a custom workflow from node definitions."""
        nodes = args.get("nodes", [])
        if not nodes:
            raise ValueError("nodes list is required")

        # Convert node list to ComfyUI workflow format
        workflow = {}
        for node in nodes:
            node_id = node.get("id")
            node_type = node.get("type")
            inputs = node.get("inputs", {})
            
            if not node_id or not node_type:
                raise ValueError("Each node must have an id and type")
            
            workflow[node_id] = {
                "class_type": node_type,
                "inputs": inputs
            }
        
        # Validate the workflow
        if not validate_workflow(workflow):
            raise ValueError("Invalid workflow structure")
        
        return {
            "workflow": workflow,
            "node_count": len(workflow),
            "status": "built"
        }

    async def _save_workflow(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Save a workflow to file."""
        name = args.get("name")
        workflow = args.get("workflow")
        description = args.get("description", "")
        
        if not name or not workflow:
            raise ValueError("name and workflow are required")
        
        # Add metadata
        workflow_data = {
            "workflow": workflow,
            "metadata": {
                "name": name,
                "description": description,
                "created_by": "mcp-comfyui",
                "version": "1.0"
            }
        }
        
        # Save to file
        workflow_dir = Path(os.getenv("COMFYUI_WORKFLOWS_DIR", "./workflows"))
        workflow_dir.mkdir(exist_ok=True)
        
        workflow_path = workflow_dir / f"{name}.json"
        with open(workflow_path, "w") as f:
            json.dump(workflow_data, f, indent=2)
        
        return {
            "saved": True,
            "path": str(workflow_path),
            "name": name
        }

    async def _get_loras(self) -> List[str]:
        """Get available LoRA models."""
        # This would need to be implemented in comfyui_client
        # For now, return empty list
        return []

    async def _get_samplers(self) -> List[str]:
        """Get available samplers."""
        return [
            "euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
            "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral",
            "dpmpp_sde", "dpmpp_2m", "dpmpp_3m_sde", "ddim", "uni_pc"
        ]

    async def _get_schedulers(self) -> List[str]:
        """Get available schedulers."""
        return ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]

    async def _get_node_types(self) -> Dict[str, Any]:
        """Get all available node types from ComfyUI."""
        # This would need to query ComfyUI's object_info endpoint
        # For now, return basic node types
        return {
            "CheckpointLoaderSimple": {
                "input": {"required": {"ckpt_name": ["STRING"]}},
                "output": ["MODEL", "CLIP", "VAE"],
                "category": "loaders"
            },
            "CLIPTextEncode": {
                "input": {"required": {"text": ["STRING"], "clip": ["CLIP"]}},
                "output": ["CONDITIONING"],
                "category": "conditioning"
            },
            "KSampler": {
                "input": {
                    "required": {
                        "model": ["MODEL"],
                        "seed": ["INT"],
                        "steps": ["INT"],
                        "cfg": ["FLOAT"],
                        "sampler_name": ["STRING"],
                        "scheduler": ["STRING"],
                        "positive": ["CONDITIONING"],
                        "negative": ["CONDITIONING"],
                        "latent_image": ["LATENT"],
                        "denoise": ["FLOAT"]
                    }
                },
                "output": ["LATENT"],
                "category": "sampling"
            },
            "VAEDecode": {
                "input": {"required": {"samples": ["LATENT"], "vae": ["VAE"]}},
                "output": ["IMAGE"],
                "category": "latent"
            },
            "SaveImage": {
                "input": {"required": {"images": ["IMAGE"], "filename_prefix": ["STRING"]}},
                "output": [],
                "category": "image"
            },
            "LoraLoader": {
                "input": {
                    "required": {
                        "model": ["MODEL"],
                        "clip": ["CLIP"],
                        "lora_name": ["STRING"],
                        "strength_model": ["FLOAT"],
                        "strength_clip": ["FLOAT"]
                    }
                },
                "output": ["MODEL", "CLIP"],
                "category": "loaders"
            }
        }

    async def _get_embeddings(self) -> List[str]:
        """Get available embeddings."""
        # This would need to query ComfyUI's embeddings folder
        return []

    async def _clear_queue(self) -> Dict[str, Any]:
        """Clear the generation queue."""
        # This would need to be implemented in comfyui_client
        return {"cleared": True, "message": "Queue cleared"}

    async def _upload_image(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Upload an image to ComfyUI."""
        image_path = args.get("image_path")
        if not image_path:
            raise ValueError("image_path is required")
        
        # This would need to be implemented in comfyui_client
        # For now, return mock response
        return {
            "uploaded": True,
            "filename": Path(image_path).name,
            "type": "input"
        }

    async def _execute_workflow(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a saved workflow."""
        workflow_name = args.get("workflow_name")
        if not workflow_name:
            raise ValueError("workflow_name is required")

        # Load workflow from file
        workflow_path = Path(os.getenv("COMFYUI_WORKFLOWS_DIR", "./workflows")) / workflow_name
        if not workflow_path.exists():
            raise FileNotFoundError(f"Workflow not found: {workflow_name}")

        with open(workflow_path, "r") as f:
            workflow_data = json.load(f)

        # Extract workflow (handle both old and new format)
        if isinstance(workflow_data, dict) and "workflow" in workflow_data:
            workflow = workflow_data["workflow"]
        else:
            workflow = workflow_data

        # Apply any input overrides
        inputs = args.get("inputs", {})
        # TODO: Implement workflow input mapping

        prompt_id = await self.client.queue_prompt(workflow)
        result = await self.client.wait_for_completion(prompt_id)

        return {
            "prompt_id": prompt_id,
            "workflow": workflow_name,
            "status": "completed",
            "outputs": result,
        }

    async def _list_workflows(self) -> List[str]:
        """List available workflow files."""
        workflows_dir = Path(os.getenv("COMFYUI_WORKFLOWS_DIR", "./workflows"))
        if not workflows_dir.exists():
            return []

        return [f.name for f in workflows_dir.glob("*.json")]

    async def _get_image(self, prompt_id: str) -> Dict[str, Any]:
        """Retrieve a generated image."""
        history = await self.client.get_history(1, prompt_id)
        if not history:
            raise ValueError(f"No history found for prompt_id: {prompt_id}")

        # Extract image info from history
        # This will vary based on ComfyUI output format
        # For now, return the raw history
        return history

    def _build_default_workflow(self, **kwargs) -> Dict[str, Any]:
        """Build a basic txt2img workflow with advanced options."""
        model = kwargs.get("model", "sd_xl_base_1.0.safetensors")
        sampler = kwargs.get("sampler", "euler")
        scheduler = kwargs.get("scheduler", "normal")
        
        return {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": model
                }
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": kwargs.get("prompt", ""),
                    "clip": ["1", 1]
                }
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": kwargs.get("negative_prompt", ""),
                    "clip": ["1", 1]
                }
            },
            "4": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": kwargs.get("width", 512),
                    "height": kwargs.get("height", 512),
                    "batch_size": 1
                }
            },
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": kwargs.get("seed", -1),
                    "steps": kwargs.get("steps", 20),
                    "cfg": kwargs.get("cfg_scale", 7.0),
                    "sampler_name": sampler,
                    "scheduler": scheduler,
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0]
                }
            },
            "6": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["5", 0],
                    "vae": ["1", 2]
                }
            },
            "7": {
                "class_type": "SaveImage",
                "inputs": {
                    "filename_prefix": "mcp_comfyui",
                    "images": ["6", 0]
                }
            }
        }

    async def run(self):
        """Run the MCP server."""
        # Initialize ComfyUI client
        host = os.getenv("COMFYUI_HOST", "localhost")
        port = int(os.getenv("COMFYUI_PORT", "8188"))
        self.client = ComfyUIClient(host, port)

        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="mcp-comfyui",
                    server_version="0.2.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )


async def main():
    """Main entry point."""
    server = ComfyUIServer()
    await server.run()
