#!/usr/bin/env python3
"""
Enhanced MCP server for ComfyUI integration with full LoRA and workflow support
"""

import asyncio
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

import httpx
from mcp.server import Server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
from pydantic import AnyUrl

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mcp-comfyui")

# Default ComfyUI API URL
COMFYUI_API_URL = os.getenv("COMFYUI_API_URL", "http://127.0.0.1:8188")


class ComfyUIClient:
    """Enhanced client for interacting with ComfyUI API"""
    
    def __init__(self, api_url: str = COMFYUI_API_URL):
        self.api_url = api_url.rstrip('/')
        self.client_id = str(uuid.uuid4())
        
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an HTTP request to ComfyUI API"""
        url = f"{self.api_url}{endpoint}"
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                response = await client.request(method, url, **kwargs)
                response.raise_for_status()
                return response.json() if response.content else {}
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error: {e}")
                raise
            except Exception as e:
                logger.error(f"Request error: {e}")
                raise
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get ComfyUI system stats"""
        return await self._make_request("GET", "/system_stats")
    
    async def get_queue(self) -> Dict[str, Any]:
        """Get current queue status"""
        return await self._make_request("GET", "/queue")
    
    async def get_history(self, limit: int = 10) -> Dict[str, Any]:
        """Get generation history"""
        return await self._make_request("GET", f"/history?max_items={limit}")
    
    async def get_models(self) -> List[str]:
        """Get available checkpoint models"""
        try:
            data = await self._make_request("GET", "/object_info")
            models = data.get("CheckpointLoaderSimple", {}).get("input", {}).get("required", {}).get("ckpt_name", [[]])[0]
            return models
        except:
            return []
    
    async def get_loras(self) -> List[str]:
        """Get available LoRA models"""
        try:
            data = await self._make_request("GET", "/object_info")
            loras = data.get("LoraLoader", {}).get("input", {}).get("required", {}).get("lora_name", [[]])[0]
            return loras
        except:
            return []
    
    async def get_vae_models(self) -> List[str]:
        """Get available VAE models"""
        try:
            data = await self._make_request("GET", "/object_info")
            vaes = data.get("VAELoader", {}).get("input", {}).get("required", {}).get("vae_name", [[]])[0]
            return vaes
        except:
            return []
    
    async def get_samplers(self) -> List[str]:
        """Get available samplers"""
        try:
            data = await self._make_request("GET", "/object_info")
            samplers = data.get("KSampler", {}).get("input", {}).get("required", {}).get("sampler_name", [[]])[0]
            return samplers
        except:
            return ["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral", 
                    "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", 
                    "dpmpp_sde", "dpmpp_2m", "dpmpp_3m_sde", "ddim", "uni_pc"]
    
    async def get_schedulers(self) -> List[str]:
        """Get available schedulers"""
        try:
            data = await self._make_request("GET", "/object_info")
            schedulers = data.get("KSampler", {}).get("input", {}).get("required", {}).get("scheduler", [[]])[0]
            return schedulers
        except:
            return ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]
    
    async def get_embeddings(self) -> Dict[str, List[str]]:
        """Get available embeddings"""
        return await self._make_request("GET", "/embeddings")
    
    async def get_node_types(self) -> Dict[str, Any]:
        """Get all available node types and their schemas"""
        return await self._make_request("GET", "/object_info")
    
    async def interrupt_generation(self) -> Dict[str, Any]:
        """Interrupt the current generation"""
        return await self._make_request("POST", "/interrupt")
    
    async def clear_queue(self) -> Dict[str, Any]:
        """Clear the generation queue"""
        return await self._make_request("POST", "/queue", json={"clear": True})
    
    async def save_workflow(self, name: str, workflow: Dict[str, Any], description: Optional[str] = None) -> Dict[str, Any]:
        """Save a workflow (stored locally for now)"""
        workflow_dir = Path("workflows")
        workflow_dir.mkdir(exist_ok=True)
        
        workflow_data = {
            "name": name,
            "description": description,
            "workflow": workflow,
            "created_at": str(asyncio.get_event_loop().time())
        }
        
        with open(workflow_dir / f"{name}.json", "w") as f:
            json.dump(workflow_data, f, indent=2)
        
        return {"status": "saved", "name": name}
    
    async def list_workflows(self) -> List[Dict[str, Any]]:
        """List saved workflows"""
        workflow_dir = Path("workflows")
        if not workflow_dir.exists():
            return []
        
        workflows = []
        for file in workflow_dir.glob("*.json"):
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                    workflows.append({
                        "name": data.get("name", file.stem),
                        "description": data.get("description", ""),
                        "created_at": data.get("created_at", "")
                    })
            except:
                continue
        
        return workflows
    
    async def execute_workflow(self, workflow: Dict[str, Any], wait: bool = True) -> Dict[str, Any]:
        """Execute a workflow"""
        # Add client_id to the workflow
        prompt = {
            "prompt": workflow,
            "client_id": self.client_id
        }
        
        # Queue the prompt
        response = await self._make_request("POST", "/prompt", json=prompt)
        prompt_id = response.get("prompt_id")
        
        if not wait or not prompt_id:
            return response
        
        # Wait for completion
        return await self._wait_for_completion(prompt_id)
    
    async def _wait_for_completion(self, prompt_id: str, timeout: float = 300.0) -> Dict[str, Any]:
        """Wait for a prompt to complete"""
        start_time = asyncio.get_event_loop().time()
        
        while True:
            if asyncio.get_event_loop().time() - start_time > timeout:
                raise TimeoutError(f"Generation timeout after {timeout} seconds")
            
            history = await self._make_request("GET", f"/history/{prompt_id}")
            
            if prompt_id in history:
                result = history[prompt_id]
                if result.get("status", {}).get("completed", False):
                    return {
                        "prompt_id": prompt_id,
                        "status": "completed",
                        "outputs": result
                    }
                elif result.get("status", {}).get("status_str") == "error":
                    return {
                        "prompt_id": prompt_id,
                        "status": "error",
                        "outputs": result
                    }
            
            await asyncio.sleep(0.5)
    
    def build_txt2img_workflow(
        self,
        prompt: str,
        negative_prompt: str = "",
        model: str = "sd_xl_base_1.0.safetensors",
        width: int = 1024,
        height: int = 1024,
        steps: int = 20,
        cfg_scale: float = 7.0,
        sampler: str = "euler",
        scheduler: str = "normal",
        seed: int = -1,
        lora_name: Optional[str] = None,
        lora_strength: float = 1.0,
        vae_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build a text-to-image workflow with optional LoRA and VAE"""
        
        workflow = {}
        node_id = 1
        
        # Checkpoint loader
        checkpoint_node = str(node_id)
        workflow[checkpoint_node] = {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": model}
        }
        model_out = [checkpoint_node, 0]
        clip_out = [checkpoint_node, 1]
        vae_out = [checkpoint_node, 2]
        node_id += 1
        
        # VAE Loader (if specified)
        if vae_name:
            vae_node = str(node_id)
            workflow[vae_node] = {
                "class_type": "VAELoader",
                "inputs": {"vae_name": vae_name}
            }
            vae_out = [vae_node, 0]
            node_id += 1
        
        # LoRA Loader (if specified)
        if lora_name:
            lora_node = str(node_id)
            workflow[lora_node] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "lora_name": lora_name,
                    "strength_model": lora_strength,
                    "strength_clip": lora_strength,
                    "model": model_out,
                    "clip": clip_out
                }
            }
            model_out = [lora_node, 0]
            clip_out = [lora_node, 1]
            node_id += 1
        
        # Positive prompt
        positive_node = str(node_id)
        workflow[positive_node] = {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": clip_out
            }
        }
        node_id += 1
        
        # Negative prompt
        negative_node = str(node_id)
        workflow[negative_node] = {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_prompt,
                "clip": clip_out
            }
        }
        node_id += 1
        
        # Empty latent
        latent_node = str(node_id)
        workflow[latent_node] = {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1
            }
        }
        node_id += 1
        
        # KSampler
        sampler_node = str(node_id)
        workflow[sampler_node] = {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed if seed >= 0 else int(uuid.uuid4().int & 0x7FFFFFFF),
                "steps": steps,
                "cfg": cfg_scale,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": 1.0,
                "model": model_out,
                "positive": [positive_node, 0],
                "negative": [negative_node, 0],
                "latent_image": [latent_node, 0]
            }
        }
        node_id += 1
        
        # VAE Decode
        decode_node = str(node_id)
        workflow[decode_node] = {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": [sampler_node, 0],
                "vae": vae_out
            }
        }
        node_id += 1
        
        # Save Image
        save_node = str(node_id)
        workflow[save_node] = {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "mcp_comfyui",
                "images": [decode_node, 0]
            }
        }
        
        return workflow


# Initialize server and client
server = Server("mcp-comfyui")
comfyui_client = ComfyUIClient()


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List all available tools"""
    return [
        Tool(
            name="generate_image",
            description="Generate an image using ComfyUI with a text prompt",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Positive prompt"},
                    "negative_prompt": {"type": "string", "description": "Negative prompt", "default": ""},
                    "model": {"type": "string", "description": "Model checkpoint to use", "default": "sd_xl_base_1.0.safetensors"},
                    "width": {"type": "integer", "description": "Image width", "default": 1024},
                    "height": {"type": "integer", "description": "Image height", "default": 1024},
                    "steps": {"type": "integer", "description": "Number of steps", "default": 20},
                    "cfg_scale": {"type": "number", "description": "CFG scale", "default": 7.0},
                    "sampler": {"type": "string", "description": "Sampler name", "default": "euler"},
                    "scheduler": {"type": "string", "description": "Scheduler type", "default": "normal"},
                    "seed": {"type": "integer", "description": "Random seed (-1 for random)", "default": -1}
                },
                "required": ["prompt"]
            }
        ),
        Tool(
            name="generate_with_lora",
            description="Generate an image using a LoRA model",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Positive prompt"},
                    "negative_prompt": {"type": "string", "description": "Negative prompt", "default": ""},
                    "model": {"type": "string", "description": "Base model checkpoint", "default": "sd_xl_base_1.0.safetensors"},
                    "lora_name": {"type": "string", "description": "LoRA model name"},
                    "lora_strength": {"type": "number", "description": "LoRA strength (0-2)", "default": 1.0},
                    "width": {"type": "integer", "description": "Image width", "default": 1024},
                    "height": {"type": "integer", "description": "Image height", "default": 1024},
                    "steps": {"type": "integer", "description": "Number of steps", "default": 20},
                    "cfg_scale": {"type": "number", "description": "CFG scale", "default": 7.0},
                    "sampler": {"type": "string", "description": "Sampler name", "default": "euler"},
                    "seed": {"type": "integer", "description": "Random seed", "default": -1}
                },
                "required": ["prompt", "lora_name"]
            }
        ),
        Tool(
            name="build_workflow",
            description="Build a custom ComfyUI workflow programmatically",
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
                                "inputs": {"type": "object"}
                            },
                            "required": ["id", "type", "inputs"]
                        }
                    }
                },
                "required": ["nodes"]
            }
        ),
        Tool(
            name="execute_workflow",
            description="Execute a saved ComfyUI workflow",
            inputSchema={
                "type": "object",
                "properties": {
                    "workflow_name": {"type": "string", "description": "Name of the workflow file"},
                    "inputs": {"type": "object", "description": "Workflow input parameters", "default": {}}
                },
                "required": ["workflow_name"]
            }
        ),
        Tool(
            name="save_workflow",
            description="Save a workflow to file",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name for the workflow"},
                    "workflow": {"type": "object", "description": "Workflow definition"},
                    "description": {"type": "string", "description": "Workflow description", "default": ""}
                },
                "required": ["name", "workflow"]
            }
        ),
        Tool(
            name="list_workflows",
            description="List available workflow files",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_queue_status",
            description="Get the current queue status",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_history",
            description="Get generation history",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Number of items to retrieve", "default": 10}
                }
            }
        ),
        Tool(
            name="get_image",
            description="Retrieve a generated image",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt_id": {"type": "string", "description": "ID of the generation"}
                },
                "required": ["prompt_id"]
            }
        ),
        Tool(
            name="list_models",
            description="List available model checkpoints",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="list_loras",
            description="List available LoRA models",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="list_samplers",
            description="List available samplers",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="list_schedulers",
            description="List available schedulers",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_node_types",
            description="Get all available node types",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_embeddings",
            description="List available embeddings",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="interrupt_generation",
            description="Interrupt the current generation",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="clear_queue",
            description="Clear the generation queue",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls"""
    
    try:
        if name == "generate_image":
            # Build and execute workflow
            workflow = comfyui_client.build_txt2img_workflow(
                prompt=arguments["prompt"],
                negative_prompt=arguments.get("negative_prompt", ""),
                model=arguments.get("model", "sd_xl_base_1.0.safetensors"),
                width=arguments.get("width", 1024),
                height=arguments.get("height", 1024),
                steps=arguments.get("steps", 20),
                cfg_scale=arguments.get("cfg_scale", 7.0),
                sampler=arguments.get("sampler", "euler"),
                scheduler=arguments.get("scheduler", "normal"),
                seed=arguments.get("seed", -1)
            )
            result = await comfyui_client.execute_workflow(workflow)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "generate_with_lora":
            # Build workflow with LoRA
            workflow = comfyui_client.build_txt2img_workflow(
                prompt=arguments["prompt"],
                negative_prompt=arguments.get("negative_prompt", ""),
                model=arguments.get("model", "sd_xl_base_1.0.safetensors"),
                width=arguments.get("width", 1024),
                height=arguments.get("height", 1024),
                steps=arguments.get("steps", 20),
                cfg_scale=arguments.get("cfg_scale", 7.0),
                sampler=arguments.get("sampler", "euler"),
                scheduler=arguments.get("scheduler", "normal"),
                seed=arguments.get("seed", -1),
                lora_name=arguments["lora_name"],
                lora_strength=arguments.get("lora_strength", 1.0)
            )
            result = await comfyui_client.execute_workflow(workflow)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "build_workflow":
            # Build custom workflow from nodes
            workflow = {}
            for node in arguments["nodes"]:
                workflow[node["id"]] = {
                    "class_type": node["type"],
                    "inputs": node["inputs"]
                }
            return [TextContent(type="text", text=json.dumps(workflow, indent=2))]
        
        elif name == "execute_workflow":
            # Load and execute saved workflow
            workflow_path = Path("workflows") / f"{arguments['workflow_name']}.json"
            if not workflow_path.exists():
                return [TextContent(type="text", text=f"Workflow not found: {arguments['workflow_name']}")]
            
            with open(workflow_path, "r") as f:
                workflow_data = json.load(f)
            
            workflow = workflow_data["workflow"]
            # Apply any input overrides
            for key, value in arguments.get("inputs", {}).items():
                if key in workflow:
                    workflow[key]["inputs"].update(value)
            
            result = await comfyui_client.execute_workflow(workflow)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "save_workflow":
            result = await comfyui_client.save_workflow(
                arguments["name"],
                arguments["workflow"],
                arguments.get("description", "")
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "list_workflows":
            workflows = await comfyui_client.list_workflows()
            return [TextContent(type="text", text=json.dumps(workflows, indent=2))]
        
        elif name == "get_queue_status":
            queue = await comfyui_client.get_queue()
            return [TextContent(type="text", text=json.dumps(queue, indent=2))]
        
        elif name == "get_history":
            history = await comfyui_client.get_history(arguments.get("limit", 10))
            return [TextContent(type="text", text=json.dumps(history, indent=2))]
        
        elif name == "get_image":
            # For now, return the path info
            return [TextContent(type="text", text=f"Images for prompt {arguments['prompt_id']} should be in ComfyUI/output/")]
        
        elif name == "list_models":
            models = await comfyui_client.get_models()
            return [TextContent(type="text", text=json.dumps(models, indent=2))]
        
        elif name == "list_loras":
            loras = await comfyui_client.get_loras()
            return [TextContent(type="text", text=json.dumps(loras, indent=2))]
        
        elif name == "list_samplers":
            samplers = await comfyui_client.get_samplers()
            return [TextContent(type="text", text=json.dumps(samplers, indent=2))]
        
        elif name == "list_schedulers":
            schedulers = await comfyui_client.get_schedulers()
            return [TextContent(type="text", text=json.dumps(schedulers, indent=2))]
        
        elif name == "get_node_types":
            node_types = await comfyui_client.get_node_types()
            # Summarize due to size
            summary = {
                "total_nodes": len(node_types),
                "categories": list(set(node.get("category", "uncategorized") for node in node_types.values() if isinstance(node, dict))),
                "sample_nodes": list(node_types.keys())[:20]
            }
            return [TextContent(type="text", text=json.dumps(summary, indent=2))]
        
        elif name == "get_embeddings":
            embeddings = await comfyui_client.get_embeddings()
            return [TextContent(type="text", text=json.dumps(embeddings, indent=2))]
        
        elif name == "interrupt_generation":
            result = await comfyui_client.interrupt_generation()
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "clear_queue":
            result = await comfyui_client.clear_queue()
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
            
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


@asynccontextmanager
async def lifespan():
    """Server lifecycle manager"""
    logger.info("Starting Enhanced MCP-ComfyUI server...")
    
    # Test connection to ComfyUI
    try:
        stats = await comfyui_client.get_system_stats()
        logger.info(f"Connected to ComfyUI: {stats}")
    except Exception as e:
        logger.warning(f"Could not connect to ComfyUI at {COMFYUI_API_URL}: {e}")
        logger.warning("Some features may not work until ComfyUI is running")
    
    yield
    
    logger.info("Shutting down Enhanced MCP-ComfyUI server...")


async def main():
    """Main entry point"""
    async with lifespan():
        from mcp.server.stdio import stdio_server
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                lifespan,
                raise_exceptions=True
            )


if __name__ == "__main__":
    asyncio.run(main())
