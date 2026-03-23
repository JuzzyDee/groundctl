"""groundctl MCP server — Ground Control to Major Claude."""

import asyncio
import base64
import json
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent

from .rover_client import RoverClient
from .navigation import WaypointNavigator

server = Server("groundctl")
rover = RoverClient()
navigator = WaypointNavigator(rover)


def _tool(name: str, description: str, properties: dict, required: list[str] | None = None) -> Tool:
    return Tool(
        name=name,
        description=description,
        inputSchema={
            "type": "object",
            "properties": properties,
            "required": required or [],
        },
    )


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        # Perception
        _tool("look", "Get front and rear camera frames from the rover. Returns images you can see directly.", {}, []),
        _tool("look_front", "Get the front camera frame only.", {}, []),
        _tool("look_rear", "Get the rear camera frame only.", {}, []),
        _tool(
            "get_status",
            "Get rover telemetry: battery, GPS (lat/lng), compass orientation, speed, signal strength, IMU data (accelerometer, gyroscope, magnetometer, wheel RPMs).",
            {},
            [],
        ),
        # Control
        _tool(
            "move",
            "Send a movement command to the rover. Linear: -1.0 (full reverse) to 1.0 (full forward). Angular: -1.0 (full left) to 1.0 (full right).",
            {
                "linear": {"type": "number", "description": "Forward/backward speed, -1.0 to 1.0"},
                "angular": {"type": "number", "description": "Turn rate, -1.0 (left) to 1.0 (right)"},
            },
            ["linear", "angular"],
        ),
        _tool("stop", "Stop the rover immediately.", {}, []),
        _tool(
            "set_lamp",
            "Turn the rover's headlights on or off.",
            {"on": {"type": "boolean", "description": "true for on, false for off"}},
            ["on"],
        ),
        _tool(
            "speak",
            "Say something through the rover's speaker using text-to-speech.",
            {"text": {"type": "string", "description": "Text to speak aloud"}},
            ["text"],
        ),
        # Missions
        _tool("start_mission", "Start a FrodoBots mission.", {}, []),
        _tool("end_mission", "End the current mission. Warning: loses all progress.", {}, []),
        _tool("get_checkpoints", "Get the list of checkpoints for the current mission.", {}, []),
        _tool("checkpoint_reached", "Report that the rover has reached a checkpoint.", {}, []),
        _tool("get_mission_history", "Get past mission records.", {}, []),
        # Navigation
        _tool(
            "navigate_to",
            "Navigate the rover to a GPS coordinate using proportional steering. The rover will continuously adjust heading and drive until it arrives within the threshold distance. This runs asynchronously — the rover drives itself while this tool is active.",
            {
                "latitude": {"type": "number", "description": "Target latitude"},
                "longitude": {"type": "number", "description": "Target longitude"},
                "speed": {"type": "number", "description": "Forward speed 0.0-1.0 (default 0.4)"},
            },
            ["latitude", "longitude"],
        ),
        _tool(
            "distance_to",
            "Calculate distance in meters from the rover's current position to a GPS coordinate.",
            {
                "latitude": {"type": "number", "description": "Target latitude"},
                "longitude": {"type": "number", "description": "Target longitude"},
            },
            ["latitude", "longitude"],
        ),
        _tool("cancel_navigation", "Cancel any active navigation. The rover will stop.", {}, []),
        # Interventions
        _tool("start_intervention", "Start an intervention for the current ride.", {}, []),
        _tool("end_intervention", "End the current intervention.", {}, []),
        _tool("get_interventions", "Get intervention history.", {}, []),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent | ImageContent]:
    match name:
        # Perception
        case "look":
            data = await rover.get_screenshot()
            content = []
            for key in ["front_frame", "rear_frame"]:
                if key in data:
                    content.append(ImageContent(
                        type="image",
                        data=data[key],
                        mimeType="image/jpeg",
                    ))
            if "timestamp" in data:
                content.append(TextContent(type="text", text=f"Timestamp: {data['timestamp']}"))
            return content or [TextContent(type="text", text="No frames available")]

        case "look_front":
            frame = await rover.get_front_frame()
            if frame:
                return [ImageContent(
                    type="image",
                    data=base64.b64encode(frame).decode(),
                    mimeType="image/jpeg",
                )]
            return [TextContent(type="text", text="Front frame not available")]

        case "look_rear":
            frame = await rover.get_rear_frame()
            if frame:
                return [ImageContent(
                    type="image",
                    data=base64.b64encode(frame).decode(),
                    mimeType="image/jpeg",
                )]
            return [TextContent(type="text", text="Rear frame not available")]

        case "get_status":
            data = await rover.get_data()
            return [TextContent(type="text", text=json.dumps(data, indent=2))]

        # Control
        case "move":
            result = await rover.move(arguments["linear"], arguments["angular"])
            return [TextContent(type="text", text=result.get("message", "OK"))]

        case "stop":
            result = await rover.stop()
            return [TextContent(type="text", text="Stopped")]

        case "set_lamp":
            result = await rover.set_lamp(arguments["on"])
            return [TextContent(type="text", text=f"Lamp {'on' if arguments['on'] else 'off'}")]

        case "speak":
            result = await rover.speak(arguments["text"])
            return [TextContent(type="text", text=result.get("message", "OK"))]

        # Missions
        case "start_mission":
            result = await rover.start_mission()
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        case "end_mission":
            result = await rover.end_mission()
            return [TextContent(type="text", text=result.get("message", "OK"))]

        case "get_checkpoints":
            result = await rover.get_checkpoints()
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        case "checkpoint_reached":
            result = await rover.checkpoint_reached()
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        case "get_mission_history":
            result = await rover.get_mission_history()
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        # Navigation
        case "navigate_to":
            speed = arguments.get("speed")
            result = await navigator.navigate_to(
                arguments["latitude"],
                arguments["longitude"],
                speed=speed,
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        case "distance_to":
            distance = await navigator.distance_to(
                arguments["latitude"], arguments["longitude"]
            )
            return [TextContent(type="text", text=f"{distance:.1f} meters")]

        case "cancel_navigation":
            await navigator.stop()
            return [TextContent(type="text", text="Navigation cancelled")]

        # Interventions
        case "start_intervention":
            result = await rover.start_intervention()
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        case "end_intervention":
            result = await rover.end_intervention()
            return [TextContent(type="text", text=result.get("message", "OK"))]

        case "get_interventions":
            result = await rover.get_interventions()
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        case _:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def run():
    asyncio.run(main())


if __name__ == "__main__":
    run()
