"""
Metrics tracking for Liar's Dice simulation.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class GameMetrics:
    """Metrics for a single game"""
    winner: str
    rounds: int
    final_lives: Dict[str, int]
    bluff_calls: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    successful_bluffs: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    failed_bluffs: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    successful_catches: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    failed_catches: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    avg_bid_quantity: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    bluff_rate: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    survival_time: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

@dataclass
class SimulationMetrics:
    """Aggregate metrics for multiple games"""
    games: List[GameMetrics] = field(default_factory=list)
    agent_types: Dict[str, str] = field(default_factory=dict)  # agent_name -> agent_type
    
    def add_game(self, metrics: GameMetrics):
        """Add a game's metrics to the simulation"""
        self.games.append(metrics)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to a pandas DataFrame for analysis"""
        data = []
        for game_idx, game in enumerate(self.games):
            for agent_name in game.final_lives.keys():
                agent_type = self.agent_types[agent_name]
                data.append({
                    'game_id': game_idx,
                    'agent_name': agent_name,
                    'agent_type': agent_type,
                    'won': agent_name == game.winner,
                    'final_lives': game.final_lives.get(agent_name, 0),
                    'bluff_calls': game.bluff_calls.get(agent_name, 0),
                    'successful_bluffs': game.successful_bluffs.get(agent_name, 0),
                    'failed_bluffs': game.failed_bluffs.get(agent_name, 0),
                    'successful_catches': game.successful_catches.get(agent_name, 0),
                    'failed_catches': game.failed_catches.get(agent_name, 0),
                    'avg_bid_quantity': game.avg_bid_quantity.get(agent_name, 0),
                    'bluff_rate': game.bluff_rate.get(agent_name, 0),
                    'survival_time': game.survival_time.get(agent_name, 0),
                    'rounds': game.rounds
                })
        return pd.DataFrame(data)
    
    def generate_reports(self, output_dir: str = 'reports'):
        """Generate comprehensive performance reports and visualizations"""
        df = self.to_dataframe()
        
        # Create output directory
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Overall Win Rates
        plt.figure(figsize=(10, 6))
        win_rates = df.groupby('agent_type')['won'].mean()
        win_rates.plot(kind='bar')
        plt.title('Win Rates by Agent Type')
        plt.ylabel('Win Rate')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/win_rates.png')
        plt.close()
        
        # 2. Survival Analysis
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='agent_type', y='survival_time')
        plt.title('Survival Time Distribution by Agent Type')
        plt.ylabel('Rounds Survived')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/survival_time.png')
        plt.close()
        
        # 3. Bluffing Behavior
        plt.figure(figsize=(12, 6))
        bluff_metrics = df.groupby('agent_type').agg({
            'bluff_rate': 'mean',
            'successful_bluffs': 'mean',
            'successful_catches': 'mean'
        }).plot(kind='bar')
        plt.title('Bluffing Behavior by Agent Type')
        plt.ylabel('Average Count')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/bluffing_behavior.png')
        plt.close()
        
        # 4. Bidding Patterns
        plt.figure(figsize=(10, 6))
        df.groupby('agent_type').agg({
            'avg_bid_quantity': 'mean'
        }).plot(kind='bar')
        plt.title('Average Bidding Patterns by Agent Type')
        plt.ylabel('Average Value')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/bidding_patterns.png')
        plt.close()
        
        # 5. Performance Over Time
        plt.figure(figsize=(12, 6))
        for agent_type in df['agent_type'].unique():
            agent_data = df[df['agent_type'] == agent_type]
            wins = agent_data.groupby('game_id')['won'].mean()
            plt.plot(wins.rolling(window=10).mean(), label=agent_type)
        plt.title('Win Rate Over Time (10-game rolling average)')
        plt.xlabel('Game Number')
        plt.ylabel('Win Rate')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/performance_over_time.png')
        plt.close()
        
        # Generate summary statistics
        summary = {
            'Overall Statistics': df.groupby('agent_type').agg({
                'won': 'mean',
                'survival_time': ['mean', 'std'],
                'bluff_rate': 'mean',
                'successful_bluffs': 'mean',
                'successful_catches': 'mean',
                'final_lives': 'mean'
            }).round(3)
        }
        
        # Save summary to CSV
        for name, stats in summary.items():
            stats.to_csv(f'{output_dir}/{name.lower().replace(" ", "_")}.csv')
