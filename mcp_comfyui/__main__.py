"""Entry point for the MCP ComfyUI server."""

import asyncio
import logging
from .server import main as server_main

logging.basicConfig(level=logging.INFO)


def main():
    """Run the MCP server."""
    asyncio.run(server_main())


if __name__ == "__main__":
    main()
