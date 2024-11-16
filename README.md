# Liar's Dice Multi-Agent Simulation

A comprehensive simulation environment for the game of Liar's Dice, featuring multiple AI agents with different strategies.

## Overview

This project implements a multi-agent simulation of Liar's Dice, a classic dice game of deception and probability. The simulation supports different types of agents, from simple random decision-makers to sophisticated probability-based strategists, to LLM-powered agents. It allows for various configurations and parameters to be set, such as the number of agents, the level of verbosity, and the number of rounds to simulate.

## Game Rules

In Liar's Dice:
- Each player starts with 5 dice and 2 lives
- Players roll their dice secretly
- Players take turns making bids about the total number of dice showing a particular value
- A bid must be higher than the previous bid (either in quantity or face value)
- Players can call "bluff" on the previous bid
- If a bluff is called:
  - If the bid was true, the challenger loses a life
  - If the bid was false, the bidder loses a life
- When a player loses all lives, they are eliminated
- Last player standing wins

## Project Structure

- `environment.py`: Core game environment and mechanics
- `agent.py`: Agent implementations (Random, Informed, LLM)
- `simulate_game.py`: Simulation runner with configurable parameters
- `liars_dice_calculator.py`: Probability calculations for informed agents

## Agent Types

1. **Random Agent**: Makes random valid moves
2. **Informed Agent**: Uses probability calculations to make decisions
3. **LLM Agent**: Uses language models for decision-making (requires API key)

## Requirements

- Python 3.x
- pandas
- scipy (for probability calculations)
- openai (for LLM agents)
- huggingface_hub (for alternative LLM providers)

## Installation

```bash
pip install pandas scipy openai huggingface_hub
```

## Usage

### Basic Game Simulation

```bash
# Run a single game with 4 random agents
python simulate_game.py --mode play --random-agents 4 --verbose 2

# Run a simulation with mixed agents
python simulate_game.py --mode sim --random-agents 2 --informed-agents 2
```

### Command Line Arguments

- `--mode`: 'play' for single game, 'sim'/'simulate' for multiple games
- `--rounds`: Number of rounds for simulation (default: 10000)
- `--verbose`: Output detail level (0=none, 1=basic, 2=detailed)
- `--random-agents`: Number of random agents
- `--informed-agents`: Number of informed agents
- `--llm-agents`: Number of LLM-based agents
- `--llm-model`: Model for LLM agents (default: 'gpt-4o')
- `--api-key`: API key for LLM agents

### Verbosity Levels

0. No output (fastest)
1. Basic output (winner only)
2. Detailed output (all moves and game state)

## Example Configurations

```bash
# Quick simulation with no output
python simulate_game.py --mode sim --rounds 1000 --verbose 0

# Detailed single game with mixed agents
python simulate_game.py --mode play --random-agents 2 --informed-agents 2 --verbose 2

# Large simulation with LLM agents
python simulate_game.py --mode sim --random-agents 2 --llm-agents 2 --api-key YOUR_KEY
```

## Output Format

The simulation provides statistics including:
- Win distribution for each agent
- Win percentages
- Total games played

For detailed output (verbose=2), each move shows:
- Current player's dice
- Bids and bluff calls
- Game state changes
- Eliminations

## Contributing

Feel free to contribute by:
1. Adding new agent types
2. Improving existing strategies
3. Enhancing the simulation environment
4. Adding analysis tools

## License

MIT License
