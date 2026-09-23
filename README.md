# My attempt №2 at creating AI for Tetris.
This projects has more complicated architecture.

## Launcher
Launcher script creates subprocesses, connects them to himself to execute safe termination. This is the entry point for you.
The programm takes the 'sims' list to decide:
1. How many simulations+agents to create
2. What modes will agents operate

### Modes
0. head AI agent. Only 1 of this type can be present in any configuration.
1. Random agent (weighted). Epsilon=1, but first 4 sims have custom weights for moves (Check ai_core.py).
2. AI-powered agent. Epsilon will destribute from 0.05 to 0.81 between all agents in this mode. For example, config [2, 2, 2, 2] will result in agents having epsilons 0.05, 0.3, 0.56, 0.81.

## AI core
This script contains main Agent class.

## AI server
This script works as main communication unit, hosting an Agent and connecting:
1. Godot simulation to self-agent, creating I/O for the main gameloop.
2. Launcher to self to process incoming termination signals.
3. IF IN MODE 0 - other Agents to self to track availability of each instance and displaying info in the Static Terminal

## Static Terminal
My custom "library" to create visuals in windows console.
