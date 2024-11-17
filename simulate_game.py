import argparse
from collections import Counter
from environment import Environment
from agent import RandomAgent, InformedAgent, AdaptiveAgent, LLMAgent
from metrics import SimulationMetrics
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Liar\'s Dice Simulation')
    
    # Game configuration
    parser.add_argument('--mode', type=str, default='play',
                      choices=['play', 'sim', 'simulate'],
                      help='Mode to run: "play" for single game, "sim"/"simulate" for multiple games')
    parser.add_argument('--rounds', type=int, default=1000,
                      help='Number of rounds to simulate in sim mode')
    parser.add_argument('--verbose', type=int, default=0,
                      choices=[0, 1, 2],
                      help='Verbosity level: 0=none, 1=basic, 2=detailed')
    parser.add_argument('--sleep-time', type=float, default=5.0,
                      help='Time to sleep between moves in verbose mode (seconds)')
    
    # Agent configuration
    parser.add_argument('--random-agents', type=int, default=0,
                      help='Number of random agents')
    parser.add_argument('--informed-agents', type=int, default=0,
                      help='Number of informed agents')
    parser.add_argument('--adaptive-agents', type=int, default=0,
                      help='Number of adaptive agents')
    parser.add_argument('--llm-agents', type=int, default=0,
                      help='Number of LLM agents')
    parser.add_argument('--llm-model', type=str, default='gpt-4o-mini',
                      help='Model to use for LLM agents')
    parser.add_argument('--api-key', type=str,
                      help='API key for LLM agents')
    
    # Output configuration
    parser.add_argument('--output-dir', type=str, default='reports',
                      help='Directory to save reports and visualizations')
    parser.add_argument('--plt-suffix', type=str, default='',
                      help='Suffix to add to plot filenames and subdirectory name')
    parser.add_argument('--clear-screen', action='store_true',
                      help='Clear screen between moves for cleaner output')
    
    return parser.parse_args()

def create_agents(args):
    agents = []
    color_idx = 0
    
    # Add random agents
    for i in range(args.random_agents):
        agents.append(RandomAgent(name=f"Random_{i+1}", color_idx=color_idx))
        color_idx += 1
    
    # Add informed agents
    for i in range(args.informed_agents):
        agents.append(InformedAgent(name=f"Informed_{i+1}", color_idx=color_idx))
        color_idx += 1
    
    # Add adaptive agents
    for i in range(args.adaptive_agents):
        agents.append(AdaptiveAgent(name=f"Adaptive_{i+1}", color_idx=color_idx))
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

def run_simulation(num_rounds: int, agents: list, verbose: int = 0, output_dir: str = 'reports', sleep_time: float = 5.0, plt_suffix: str = '', clear_screen: bool = False):
    # Initialize metrics tracking
    metrics = SimulationMetrics()
    metrics.verbose = verbose  # Set verbosity level
    
    # Record agent types
    for agent in agents:
        agent_type = metrics._get_agent_type(agent.__class__.__name__)
        if verbose >= 1:
            print(f"Recording agent: {agent.name} -> {agent_type}")
        metrics.agent_types[agent.name] = agent_type
    
    env = Environment(agents, verbose=verbose, sleep_time=sleep_time, clear_screen=clear_screen)
    
    for i in range(num_rounds):
        if verbose >= 1 and (i + 1) % 10 == 0:
            print(f"Round {i+1}/{num_rounds}")
        
        # Set current game number
        env.game_number = i
        
        # Play game and collect metrics
        winner = env.play_game()
        game_metrics = env.get_metrics()
        metrics.add_game(game_metrics)
        
        # Reset for next game
        env.reset()
    
    # Generate comprehensive reports
    metrics.generate_reports(output_dir, plt_suffix)
    
    # Print summary
    # if verbose >= 1:
    #     df = metrics.to_dataframe()
    #     print("\nSimulation Results:")
    #     print(f"Total games: {num_rounds}")
        
    #     print("\nWin Distribution:")
    #     for agent_type in df['agent_type'].unique():
    #         wins = df[df['agent_type'] == agent_type]['won'].sum()
    #         win_rate = (wins / num_rounds) * 100
    #         print(f"{agent_type}:")
    #         print(f"  Wins: {wins} ({win_rate:.2f}%)")
    #         print(f"  Avg Survival Time: {df[df['agent_type'] == agent_type]['survival_time'].mean():.1f} rounds")
    #         print(f"  Bluff Success Rate: {df[df['agent_type'] == agent_type]['successful_bluffs'].mean():.2f}")

def main():
    args = parse_args()
    
    # Create agents based on configuration
    agents = create_agents(args)
    
    if len(agents) < 2:
        print("Error: Need at least 2 agents to play!")
        return
    
    if args.mode in ['sim', 'simulate']:
        run_simulation(args.rounds, agents, args.verbose, args.output_dir, 
                      sleep_time=args.sleep_time, plt_suffix=args.plt_suffix,
                      clear_screen=args.clear_screen)
    else:
        # Single game mode
        env = Environment(agents, verbose=args.verbose, sleep_time=args.sleep_time, clear_screen=args.clear_screen)
        env.play_game()
        
        # Even in play mode, collect and show metrics
        if args.verbose >= 1:
            game_metrics = env.get_metrics()
            print("\nGame Statistics:")
            print(f"Total Rounds: {game_metrics.rounds}")
            print("\nBluffing Stats:")
            for agent in agents:
                print(f"\n{agent.name}:")
                print(f"  Successful Bluffs: {game_metrics.successful_bluffs[agent.name]}")
                print(f"  Successful Catches: {game_metrics.successful_catches[agent.name]}")
                print(f"  Bluff Rate: {game_metrics.bluff_rate[agent.name]:.2%}")
        
        print("\nGame complete.")

if __name__ == "__main__":
    main()