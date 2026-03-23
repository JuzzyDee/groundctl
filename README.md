# groundctl

Ground Control to Major Claude — autonomous navigation for EarthRover.

> **Status: Scaffolding.** This project is in early development. Nothing works yet. The rover hasn't even arrived. But the architecture is real and the roadmap is concrete.

## What this is

An MCP server that gives Claude eyes, wheels, and a voice through a [FrodoBots EarthRover](https://frodobots.com/). Not remote control — autonomous navigation with Claude as mission control.

## What this is not

This is not "hand Claude the API and let it send move commands." Direct API control produces lurching stop-start movement because an LLM can't issue motor commands at the frequency smooth driving requires.

groundctl is building toward a three-layer autonomy stack where Claude does what LLMs are good at (reasoning, planning, decision-making) and trained models do what they're good at (real-time reactive control).

## Architecture

```
┌─────────────────────────────────────┐
│          Claude (Mission Control)    │
│  Strategic decisions, mode switching │
│  "Follow that path" / "Go look at X"│
└──────────────┬──────────────────────┘
               │ MCP tools
┌──────────────▼──────────────────────┐
│          groundctl (Control Layer)   │
│  Mode manager, model orchestration,  │
│  command translation                 │
├──────────┬───────────┬──────────────┤
│ CV Model │ IL Model  │ Beacon       │
│ (YOLO)   │ (Driving) │ (GPS track)  │
│ Scene    │ Smooth    │ Follow       │
│ under-   │ motor     │ phone        │
│ standing │ control   │ signal       │
└──────────┴─────┬─────┴──────────────┘
                 │ SDK REST API
┌────────────────▼────────────────────┐
│          EarthRover Mini+           │
│  Cameras, GPS, IMU, Speaker, 4G    │
└─────────────────────────────────────┘
```

**Claude (prefrontal cortex):** Sees through cameras, reasons about the environment, issues high-level commands. Operates at human decision-making pace. Doesn't need to be fast.

**CV model (eyes):** Pre-trained object detection (YOLO). Identifies and locates objects in camera frames — people, dogs, obstacles, paths. Gives Claude structured scene understanding with coordinates, not just pixels.

**IL model (motor cortex):** Imitation learning model trained on driving data. Takes camera frames + high-level direction, outputs smooth continuous motor commands. Handles obstacle avoidance, path following, and steering at frame rate. The part that needs to be fast.

**Beacon (GPS tracking):** iOS app broadcasts phone location. The rover can follow the beacon signal for hands-free companion mode — walk with your dog, the rover follows along.

Three input modes, switchable on the fly:
1. **Vision tracking** — follow a bounding box (person, object)
2. **Beacon following** — follow a GPS signal (phone)
3. **Autonomous navigation** — go to a coordinate or object

## Roadmap

### Phase 1: MCP Server (now)
Direct tool access to the rover SDK. Claude can see through cameras, read telemetry, send movement commands, and speak through the speaker. Human acts as the motor control layer — Claude gives directions, human drives.

### Phase 2: Data Collection
Human drives the rover, collecting paired camera/control data for local training. FrodoBots gamifies this — you earn points for driving. Local Yeppoon data combined with the [FrodoBots-2K dataset](https://huggingface.co/datasets/frodobots/FrodoBots-2K) (2,000 hours of real-world driving data from 10+ cities).

### Phase 3: Imitation Learning Model
Behavioral cloning trained on the combined dataset. Camera frame in, motor commands out. Trained on an RTX 4090, deployed to M3 Max for inference. The rover can now drive smoothly without human input.

### Phase 4: Computer Vision Integration
YOLO object detection gives Claude structured scene understanding. Object tracking (ByteTrack) for continuous following. Distance estimation from bounding box size. The rover can now "see" objects and follow them.

### Phase 5: Beacon + Mode Switching
iOS companion app for GPS beacon. Claude switches between vision tracking, beacon following, and autonomous navigation depending on the situation. The full stack.

## Current State

The MCP server skeleton is complete with 16 tools:

| Category | Tools |
|----------|-------|
| Perception | `look`, `look_front`, `look_rear`, `get_status` |
| Control | `move`, `stop`, `set_lamp`, `speak` |
| Missions | `start_mission`, `end_mission`, `get_checkpoints`, `checkpoint_reached`, `get_mission_history` |
| Interventions | `start_intervention`, `end_intervention`, `get_interventions` |

## Training Data

- [FrodoBots-2K](https://huggingface.co/datasets/frodobots/FrodoBots-2K) — 2,000 hours of teleoperated driving data, CC-BY-SA-4.0
- Local collection from Yeppoon, QLD for Australian terrain and conditions

## Requirements

- Python 3.11+
- [EarthRover SDK](https://github.com/nicejobinc/earth-rovers-sdk) running locally
- EarthRover Mini+ (or compatible FrodoBots rover)

## License

MIT
