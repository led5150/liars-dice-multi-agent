import argparse
from collections import Counter
from environment import Environment
from agent import RandomAgent, InformedAgent, LLMAgent

def parse_args():
    parser = argparse.ArgumentParser(description='Liar\'s Dice Simulation')
    parser.add_argument('--mode', type=str, default='play',
                      choices=['play', 'sim', 'simulate'],
                      help='Mode to run: "play" for single game, "sim"/"simulate" for multiple games')
    parser.add_argument('--rounds', type=int, default=10000,
                      help='Number of rounds to simulate in sim mode')
    parser.add_argument('--verbose', type=int, default=0,
                      choices=[0, 1, 2],
                      help='Verbosity level: 0=none, 1=basic, 2=detailed')
    
    # Agent configuration
    parser.add_argument('--random-agents', type=int, default=0,
                      help='Number of random agents')
    parser.add_argument('--informed-agents', type=int, default=0,
                      help='Number of informed agents')
    parser.add_argument('--llm-agents', type=int, default=0,
                      help='Number of LLM agents')
    parser.add_argument('--llm-model', type=str, default='gpt-3.5-turbo',
                      help='Model to use for LLM agents')
    parser.add_argument('--api-key', type=str,
                      help='API key for LLM agents')
    
    return parser.parse_args()

def create_agents(args):
    agents = []
    color_idx = 0

    # make sure we have at least 2 agents
    if args.random_agents + args.informed_agents + args.llm_agents < 2:
        raise ValueError("Need at least 2 agents to play!")
    
    # Add random agents
    for i in range(args.random_agents):
        agents.append(RandomAgent(name=f"Random_{i+1}", color_idx=color_idx))
        color_idx += 1
    
    # Add informed agents
    for i in range(args.informed_agents):
        agents.append(InformedAgent(name=f"Informed_{i+1}", color_idx=color_idx))
        color_idx += 1
    
    # Add LLM agents
    for i in range(args.llm_agents):
        agents.append(LLMAgent(
            name=f"LLM_{i+1}",
            color_idx=color_idx,
            model=args.llm_model,
            api_key=args.api_key
        ))
        color_idx += 1
    
    return agents

def run_simulation(num_rounds: int, agents: list, verbose: int = 0):
    # Track wins for each agent
    wins = Counter()
    env = Environment(agents, verbose=verbose)
    
    for i in range(num_rounds):
        if verbose >= 1 and (i + 1) % 1000 == 0:
            print(f"Round {i+1}/{num_rounds}")
        
        winner = env.play_game()
        wins[winner.name] += 1
        env.reset()
    
    # Print statistics
    print("\nSimulation Results:")
    print(f"Total games: {num_rounds}")
    print("\nWin Distribution:")
    for agent_name, win_count in wins.items():
        win_percentage = (win_count / num_rounds) * 100
        print(f"{agent_name}: {win_count} wins ({win_percentage:.2f}%)")

def main():
    args = parse_args()
    
    # Create agents based on configuration
    agents = create_agents(args)
    
    if len(agents) < 2:
        print("Error: Need at least 2 agents to play!")
        return
    
    if args.mode in ['sim', 'simulate']:
        run_simulation(args.rounds, agents, args.verbose)
    else:
        # Single game mode
        env = Environment(agents, verbose=args.verbose)
        env.play_game()
        print("\nGame complete.")

if __name__ == "__main__":
    main()