#!/usr/bin/env python3
"""Example script for generating Crisis Corps logos using MCP ComfyUI."""

import asyncio
import json
from pathlib import Path

# This example assumes you have the MCP client set up
# In practice, you'd use this through Claude Desktop

# Example prompts for Crisis Corps branding
LOGO_PROMPTS = [
    # Main logo concepts
    {
        "prompt": "Crisis Corps logo, heroic robot silhouette inside shield, orange and blue, minimalist design, emergency response emblem",
        "style": "main_logo"
    },
    {
        "prompt": "CC monogram logo, interlocking letters, tech startup style, orange accent on blue, clean geometric design",
        "style": "monogram"
    },
    {
        "prompt": "Robot head icon, friendly but strong, circular badge design, Crisis Corps mascot, orange LED eyes",
        "style": "mascot"
    },
    # App icons
    {
        "prompt": "mobile app icon, Crisis Corps, rounded square, flat design, robot symbol, orange and blue gradient",
        "style": "app_icon",
        "size": 512
    },
    # Social media variations
    {
        "prompt": "Crisis Corps social media profile picture, circular design, robot emblem, high contrast, memorable icon",
        "style": "social_avatar",
        "size": 400
    },
    # Banner/header designs
    {
        "prompt": "Crisis Corps banner logo, horizontal layout, robot army silhouettes, emergency orange skyline, professional header",
        "style": "banner",
        "width": 1920,
        "height": 480
    },
    # Gaming elements
    {
        "prompt": "Crisis Corps game logo, epic style, metallic text with robot guardian, action game branding, orange energy effects",
        "style": "game_logo"
    },
    # Badge/achievement designs
    {
        "prompt": "Crisis Corps achievement badge, bronze rank, hexagonal shape, robot emblem, gamification element",
        "style": "badge_bronze"
    },
    {
        "prompt": "Crisis Corps achievement badge, gold rank, star shape, elite robot emblem, premium game asset",
        "style": "badge_gold"
    },
]

# Color variations to test
COLOR_SCHEMES = [
    "orange and blue",  # Primary
    "emergency orange and steel grey",  # Alternative 1
    "bright orange and deep navy",  # Alternative 2
    "sunset orange and electric blue",  # Vibrant option
]

# Style modifiers for different use cases
STYLE_MODIFIERS = {
    "corporate": "professional, clean, trustworthy, Fortune 500 style",
    "gaming": "dynamic, exciting, AAA game quality, epic",
    "humanitarian": "hopeful, helping hands, global unity, care",
    "tech": "futuristic, innovative, cutting-edge, Silicon Valley",
    "military": "tactical, organized, disciplined, mission-ready",
}


async def generate_logo_variations():
    """Generate multiple logo variations for Crisis Corps."""
    
    # Example of how you'd structure the generation
    variations = []
    
    # Generate main logo with different color schemes
    for color_scheme in COLOR_SCHEMES:
        prompt = f"Crisis Corps logo, minimalist robot head in shield shape, {color_scheme}, vector style, scalable design"
        variations.append({
            "prompt": prompt,
            "name": f"main_logo_{color_scheme.replace(' ', '_')}",
            "width": 1024,
            "height": 1024,
        })
    
    # Generate style variations
    for style_name, style_modifier in STYLE_MODIFIERS.items():
        prompt = f"Crisis Corps logo, {style_modifier}, robot symbol, orange and blue"
        variations.append({
            "prompt": prompt,
            "name": f"style_{style_name}",
            "width": 1024,
            "height": 1024,
        })
    
    return variations


# Prompt templates for different logo needs
LOGO_TEMPLATES = {
    "main": "Crisis Corps logo, {style} design, robot {element}, {colors}, {mood}",
    "text": "CRISIS CORPS text logo, {font_style} typography, {effects}, {colors}",
    "icon": "Crisis Corps icon, {shape} shape, {style}, {size}px, app icon design",
    "emblem": "Crisis Corps emblem, {military_style} insignia, robot {feature}, metallic finish",
    "mascot": "Crisis Corps mascot, friendly robot character, {personality}, {pose}, branded colors",
}


if __name__ == "__main__":
    # This would be called through MCP in practice
    print("Crisis Corps Logo Generation Examples")
    print("====================================\n")
    
    print("Available Workflows:")
    print("- logo_generator.json: General logo creation")
    print("- crisis_corps_logo.json: Specific Crisis Corps branding")
    print("- robot_emblem.json: Military-style emblems")
    print("- text_logo_variations.json: Typography focused\n")
    
    print("Example Prompts:")
    for i, prompt_data in enumerate(LOGO_PROMPTS[:5], 1):
        print(f"{i}. {prompt_data['prompt']}")
    
    print("\nTo use with MCP:")
    print("1. Execute a workflow: execute_workflow('crisis_corps_logo.json')")
    print("2. Generate custom: generate_image(prompt='your custom prompt', width=1024, height=1024)")
    print("3. Batch generate: Loop through LOGO_PROMPTS with different seeds")
