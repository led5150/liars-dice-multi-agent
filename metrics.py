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
import colorsys
from reasoning_analysis import ReasoningAnalyzer
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    elimination_order: List[str] = field(default_factory=list)
    
    # New metrics for enhanced analysis
    prev_bid_quantity_successful_bluff: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))
    prev_bid_quantity_failed_bluff: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))
    predicted_probability: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    actual_outcome: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    bluff_threshold: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    predicted_bluff_rate: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    actual_bluff_rate: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    decision_type: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    successful_move: Dict[str, List[bool]] = field(default_factory=lambda: defaultdict(list))

@dataclass
class SimulationMetrics:
    """Aggregate metrics for multiple games"""
    games: List[GameMetrics] = field(default_factory=list)
    agent_types: Dict[str, str] = field(default_factory=dict)  # agent_name -> agent_type
    verbose: int = 0  # Add verbose attribute
    
    def _get_agent_type(self, agent_class_name: str) -> str:
        """
        Convert agent class name to standardized agent type string.
        Agent types are stored without the 'Agent' suffix for cleaner display.
        
        Args:
            agent_class_name: Full class name (e.g., 'RandomAgent', 'InformedAgent')
            
        Returns:
            Standardized agent type (e.g., 'Random', 'Informed')
        """
        return agent_class_name.replace('Agent', '')
    
    def add_game(self, metrics: GameMetrics):
        """Add a game's metrics to the simulation"""
        self.games.append(metrics)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to a pandas DataFrame."""
        data = []
        for game_id, game in enumerate(self.games):
            # Add winner data
            data.append({
                'game_id': game_id,
                'agent_name': game.winner,
                'agent_type': self.agent_types[game.winner],
                'won': 1,
                # Basic metrics
                'final_lives': game.final_lives.get(game.winner, 0),
                'bluff_calls': game.bluff_calls.get(game.winner, 0),
                'successful_bluffs': game.successful_bluffs.get(game.winner, 0),
                'failed_bluffs': game.failed_bluffs.get(game.winner, 0),
                'successful_catches': game.successful_catches.get(game.winner, 0),
                'failed_catches': game.failed_catches.get(game.winner, 0),
                'avg_bid_quantity': game.avg_bid_quantity.get(game.winner, 0),
                'bluff_rate': game.bluff_rate.get(game.winner, 0),
                'survival_time': game.survival_time.get(game.winner, 0),
                'elimination_order': game.elimination_order,
                
                # List-type metrics
                'prev_bid_quantity_successful_bluff': game.prev_bid_quantity_successful_bluff.get(game.winner, []),
                'prev_bid_quantity_failed_bluff': game.prev_bid_quantity_failed_bluff.get(game.winner, []),
                'predicted_probability': game.predicted_probability.get(game.winner, []),
                'actual_outcome': game.actual_outcome.get(game.winner, []),
                'bluff_threshold': game.bluff_threshold.get(game.winner, []),
                'predicted_bluff_rate': game.predicted_bluff_rate.get(game.winner, []),
                'actual_bluff_rate': game.actual_bluff_rate.get(game.winner, []),
                'decision_type': game.decision_type.get(game.winner, []),
                'successful_move': game.successful_move.get(game.winner, [])
            })
            
            # Add data for non-winners
            for agent_name in self.agent_types.keys():
                if agent_name != game.winner:
                    data.append({
                        'game_id': game_id,
                        'agent_name': agent_name,
                        'agent_type': self.agent_types[agent_name],
                        'won': 0,
                        # Basic metrics
                        'final_lives': game.final_lives.get(agent_name, 0),
                        'bluff_calls': game.bluff_calls.get(agent_name, 0),
                        'successful_bluffs': game.successful_bluffs.get(agent_name, 0),
                        'failed_bluffs': game.failed_bluffs.get(agent_name, 0),
                        'successful_catches': game.successful_catches.get(agent_name, 0),
                        'failed_catches': game.failed_catches.get(agent_name, 0),
                        'avg_bid_quantity': game.avg_bid_quantity.get(agent_name, 0),
                        'bluff_rate': game.bluff_rate.get(agent_name, 0),
                        'survival_time': game.survival_time.get(agent_name, 0),
                        'elimination_order': game.elimination_order,
                        
                        # List-type metrics
                        'prev_bid_quantity_successful_bluff': game.prev_bid_quantity_successful_bluff.get(agent_name, []),
                        'prev_bid_quantity_failed_bluff': game.prev_bid_quantity_failed_bluff.get(agent_name, []),
                        'predicted_probability': game.predicted_probability.get(agent_name, []),
                        'actual_outcome': game.actual_outcome.get(agent_name, []),
                        'bluff_threshold': game.bluff_threshold.get(agent_name, []),
                        'predicted_bluff_rate': game.predicted_bluff_rate.get(agent_name, []),
                        'actual_bluff_rate': game.actual_bluff_rate.get(agent_name, []),
                        'decision_type': game.decision_type.get(agent_name, []),
                        'successful_move': game.successful_move.get(agent_name, [])
                    })
        
        # Create DataFrame and add debug info if verbose
        df = pd.DataFrame(data)
        # if self.verbose >= 2:
        #     print("\nDataFrame Info:")
        #     print(df.info())
        #     print("\nSample of list-type metrics:")
        #     list_cols = ['prev_bid_quantity_successful_bluff', 'prev_bid_quantity_failed_bluff',
        #                 'predicted_probability', 'actual_outcome', 'bluff_threshold',
        #                 'predicted_bluff_rate', 'actual_bluff_rate', 'decision_type',
        #                 'successful_move']
        #     print(df[list_cols].head())
        
        return df

    def get_plot_colors(self):
        """Get consistent plot colors across all plots."""
        # Set style
        plt.style.use('bmh')
        
        # Get pastel colors and darken them
        base_colors = sns.color_palette("pastel")
        colors = []
        for color in base_colors:
            hsv = list(colorsys.rgb_to_hsv(*color))
            hsv[1] = min(1.0, hsv[1] * 1.5)  # Increase saturation
            hsv[2] = hsv[2] * 0.8  # Decrease value (darken)
            darkened_color = colorsys.hsv_to_rgb(*hsv)
            colors.append((*darkened_color, 0.9))
            
        # Set default figure properties
        plt.rcParams['figure.facecolor'] = '#e6e6e6'
        plt.rcParams['axes.facecolor'] = '#e0e0e0'
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.4
        plt.rcParams['grid.color'] = '#666666'
        
        return colors

    def plot_win_rates(self, output_dir: str, plt_suffix: str):
        """Plot win rates by agent type."""
        df = self.to_dataframe()
        
        # First, get total games per agent type
        games_per_type = df.groupby('agent_type').size()
        
        # Get wins per agent type
        wins_per_type = df[df['won'] == 1].groupby('agent_type').size()
        
        # Calculate win rate as percentage
        win_rates = (wins_per_type / games_per_type * 100).fillna(0)
        
        # Sort by agent types
        agent_types = sorted(df['agent_type'].unique())
        win_rates = win_rates.reindex(agent_types)
        
        # Plot
        plt.figure(figsize=(10, 6))
        colors = self.get_plot_colors()
        ax = win_rates.plot(kind='bar', color=[colors[i] for i in range(len(agent_types))])
        plt.title('Win Rates by Agent Type', pad=20, fontsize=12, fontweight='bold')
        plt.ylabel('Win Rate (%)')
        
        # Add value labels on bars
        for i, v in enumerate(win_rates):
            ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/win_rates{plt_suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_survival_time(self, output_dir: str, plt_suffix: str):
        """Plot survival time distribution by agent type."""
        df = self.to_dataframe()
        agent_types = sorted(df['agent_type'].unique())
        colors = self.get_plot_colors()
        suffix = f"_{plt_suffix}" if plt_suffix else ""
        
        plt.figure(figsize=(8, 6))
        
        sorted_df = df.copy()
        sorted_df['agent_type'] = pd.Categorical(sorted_df['agent_type'], categories=agent_types)
        
        # Create color mapping for consistent colors
        color_dict = {agent: colors[i] for i, agent in enumerate(agent_types)}
        
        sns.boxplot(data=sorted_df, x='agent_type', y='survival_time', hue='agent_type', 
                   palette=color_dict, legend=False)
        plt.title('Survival Time Distribution by Agent Type', pad=20, fontsize=12, fontweight='bold')
        plt.ylabel('Rounds Survived')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/survival_time{suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_bluffing_behavior(self, output_dir: str, plt_suffix: str):
        """Plot bluffing behavior metrics by agent type."""
        df = self.to_dataframe()
        agent_types = sorted(df['agent_type'].unique())
        colors = self.get_plot_colors()
        suffix = f"_{plt_suffix}" if plt_suffix else ""
        
        # Increase figure width to accommodate legend
        fig, ax = plt.subplots(figsize=(14, 8))
        
        metrics_df = df.groupby('agent_type').agg({
            'bluff_rate': 'mean',
            'successful_bluffs': 'mean',
            'failed_bluffs': 'mean',
            'successful_catches': 'mean',
            'failed_catches': 'mean'
        })
        metrics_df = metrics_df.reindex(agent_types)
        
        bar_width = 0.15
        x = np.arange(len(agent_types))
        
        # Plot bars
        metrics = ['bluff_rate', 'successful_bluffs', 'failed_bluffs', 'successful_catches', 'failed_catches']
        labels = ['Call Bluff Rate (%)', 'Successful Bluffs/Game', 'Failed Bluffs/Game', 'Successful Catches/Game', 'Failed Catches/Game']
        
        for i, (metric, label) in enumerate(zip(metrics, labels)):
            offset = (i - 1.5) * bar_width
            # Use raw values for plotting (keep bluff_rate as decimal)
            bars = ax.bar(x + offset, metrics_df[metric], bar_width,
                         label=label, color=plt.cm.Set2(i/4))
            
            # Add value labels on bars
            for j, bar in enumerate(bars):
                height = bar.get_height()
                # Format as percentage for bluff rate, regular number for others
                if metric == 'bluff_rate':
                    value_text = f'{height * 100:.1f}%'  # Convert to percentage only for display
                else:
                    value_text = f'{height:.1f}'
                ax.text(bar.get_x() + bar_width/2, height + 0.01,
                       value_text, ha='center', va='bottom',
                       fontsize=8, fontweight='bold')
        
        # Customize plot
        ax.set_title('Bluffing Behavior by Agent Type', pad=20, fontsize=12, fontweight='bold')
        ax.set_xlabel('Agent Type')
        ax.set_ylabel('Average per Game')
        
        # Set x-axis labels
        ax.set_xticks(x)
        ax.set_xticklabels(agent_types)
        
        # Add legend
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                 frameon=True, facecolor='white', framealpha=1)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/bluffing_behavior{suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_bid_patterns(self, output_dir: str, plt_suffix: str):
        """Plot bidding patterns by agent type."""
        df = self.to_dataframe()
        
        if df.empty:
            if self.verbose >= 1:
                print("Warning: No data available for bid patterns plot")
            return
        
        plt.figure(figsize=(12, 6))
        colors = self.get_plot_colors()
        
        # Switch to bar plot for more reliable visualization
        agent_types = sorted(df['agent_type'].unique())
        bid_means = []
        bid_stds = []
        
        for agent_type in agent_types:
            agent_data = df[df['agent_type'] == agent_type]
            bid_means.append(agent_data['avg_bid_quantity'].mean())
            bid_stds.append(agent_data['avg_bid_quantity'].std())
        
        x = np.arange(len(agent_types))
        bars = plt.bar(x, bid_means, yerr=bid_stds, capsize=5,
                      color=[colors[i] for i in range(len(agent_types))],
                      alpha=0.7, label=agent_types)
        
        plt.xticks(x, agent_types, rotation=45)
        plt.title('Average Bid Quantities by Agent Type')
        plt.xlabel('Agent Type')
        plt.ylabel('Average Bid Quantity')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars with offset to avoid error bars
        for i, v in enumerate(bid_means):
            # Calculate total height including error bar
            total_height = v + bid_stds[i]
            # Add label with small horizontal offset to avoid error bar
            plt.text(i + 0.1, total_height + 0.05, f'{v:.2f}', 
                    ha='center', va='bottom')
        
        # Only add legend if we have bars
        if len(bars) > 0:
            plt.legend(title='Agent Type')
        
        suffix = f"_{plt_suffix}" if plt_suffix else ""
        plt.savefig(f'{output_dir}/bid_patterns{suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_win_progression(self, output_dir: str, plt_suffix: str):
        """Plot win rate progression over time."""
        df = self.to_dataframe()
        agent_types = sorted(df['agent_type'].unique())
        colors = self.get_plot_colors()
        suffix = f"_{plt_suffix}" if plt_suffix else ""
        
        plt.figure(figsize=(10, 6))
        for i, agent_type in enumerate(agent_types):
            agent_data = df[df['agent_type'] == agent_type]
            wins = agent_data.groupby('game_id')['won'].mean()
            cumulative_wins = wins.expanding().mean() * 100
            plt.plot(cumulative_wins.index, cumulative_wins.values, 
                    label=agent_type, color=colors[i],
                    linewidth=2, marker='o', markersize=4)
        
        plt.title('Win Rate Over Time (Cumulative)', pad=20, fontsize=12, fontweight='bold')
        plt.xlabel('Game Number')
        plt.ylabel('Win Rate (%)')
        plt.legend(title='Agent Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/performance_over_time{suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_elimination_distribution(self, output_dir: str, plt_suffix: str):
        """Plot elimination order distribution."""
        df = self.to_dataframe()
        agent_types = sorted(df['agent_type'].unique())
        colors = self.get_plot_colors()
        suffix = f"_{plt_suffix}" if plt_suffix else ""
        
        plt.figure(figsize=(10, 6))
        
        elimination_stats = {agent_type: {'first': 0, 'second': 0, 'third': 0, 'survived': 0} 
                           for agent_type in agent_types}
        
        for _, row in df.iterrows():
            agent_type = row['agent_type']
            elim_order = row['elimination_order']
            
            if row['won'] == 1:
                elimination_stats[agent_type]['survived'] += 1
            else:
                try:
                    pos = elim_order.index(row['agent_name'])
                    if pos == 0:
                        elimination_stats[agent_type]['first'] += 1
                    elif pos == 1:
                        elimination_stats[agent_type]['second'] += 1
                    elif pos == 2:
                        elimination_stats[agent_type]['third'] += 1
                except ValueError:
                    continue
        
        data = []
        for agent_type in agent_types:
            data.append({
                'Agent Type': agent_type,
                'Survived': elimination_stats[agent_type]['survived'],
                'Third': elimination_stats[agent_type]['third'],
                'Second': elimination_stats[agent_type]['second'],
                'First': elimination_stats[agent_type]['first']
            })
        
        elim_df = pd.DataFrame(data).set_index('Agent Type')
        
        ax = elim_df.plot(
            kind='bar',
            stacked=True,
            color=[colors[i] for i in range(4)],
            width=0.8
        )
        
        plt.title('Elimination Order Distribution by Agent Type', pad=20, fontsize=12, fontweight='bold')
        plt.xlabel('Agent Type')
        plt.ylabel('Number of Games')
        
        for c in ax.containers:
            ax.bar_label(c, fmt='%d', label_type='center', fontsize=8)
        
        plt.xticks(range(len(agent_types)), agent_types, rotation=0)
        
        plt.legend(
            title='Position',
            loc='center left',
            bbox_to_anchor=(1.0, 0.5),
            fontsize=8,
            title_fontsize=9
        )
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/elimination_distribution{suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_individual_agent_metrics(self, output_dir: str = 'reports', plt_suffix: str = ''):
        """Plot metrics for individual agents rather than aggregating by type."""
        df = self.to_dataframe()
        
        # Get unique agent names and their types for coloring
        agent_names = sorted(df['agent_name'].unique())
        agent_types = [self.agent_types[name] for name in agent_names]
        
        # Create color map for each agent type
        unique_types = sorted(set(agent_types))
        type_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))
        color_map = dict(zip(unique_types, type_colors))
        colors = [color_map[agent_type] for agent_type in agent_types]
        
        # Calculate per-agent metrics
        total_games = len(self.games)  # Get total number of games
        metrics_df = df.groupby('agent_name').agg({
            'won': lambda x: sum(x) / total_games,  # Keep as decimal for plotting
            'successful_bluffs': 'mean',  # Average successful bluffs per game
            'failed_bluffs': 'mean',  # Average failed bluffs per game
            'successful_catches': 'mean'  # Average successful catches per game
        })
        metrics_df = metrics_df.reindex(agent_names)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(15, 8))
        bar_width = 0.2
        x = np.arange(len(agent_names))
        
        # Plot each metric
        metrics = ['won', 'successful_bluffs', 'failed_bluffs', 'successful_catches']
        labels = ['Win Rate (%)', 'Successful Bluffs/Game', 'Failed Bluffs/Game', 'Successful Catches/Game']
        
        for i, (metric, label) in enumerate(zip(metrics, labels)):
            offset = (i - 1.5) * bar_width
            bars = ax.bar(x + offset, metrics_df[metric], bar_width,
                         label=label, color=plt.cm.Set2(i/4))
            
            # Add value labels on bars
            for j, bar in enumerate(bars):
                height = bar.get_height()
                # Add % symbol and convert to percentage for win rate
                if metric == 'won':
                    value_text = f'{height * 100:.1f}%'
                else:
                    value_text = f'{height:.1f}'
                ax.text(bar.get_x() + bar_width/2, height + 0.01,
                       value_text, ha='center', va='bottom',
                       fontsize=8, fontweight='bold')
        
        # Customize the plot
        ax.set_title('Individual Agent Performance Metrics', pad=20, fontsize=12, fontweight='bold')
        ax.set_ylabel('Value')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{name}\n({self.agent_types[name]})' for name in agent_names],
                          rotation=45, ha='right')
        
        # Add legend
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                 frameon=True, facecolor='white', framealpha=1)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/individual_agent_metrics{plt_suffix}.png',
                   bbox_inches='tight', dpi=300)
        plt.close()

    def plot_individual_win_progression(self, output_dir: str, plt_suffix: str):
        """Plot win rate progression over time for individual agents."""
        df = self.to_dataframe()
        agent_names = sorted(df['agent_name'].unique())
        agent_types = [self.agent_types[name] for name in agent_names]
        
        # Create a unique color for each agent using a mix of colormaps
        n_agents = len(agent_names)
        
        # Use multiple colormaps to get more distinct colors
        cmaps = [plt.cm.Set1, plt.cm.Set2, plt.cm.Set3, plt.cm.Paired]
        colors = []
        for i in range(n_agents):
            cmap = cmaps[i // 8]  # Switch colormaps every 8 colors
            color = cmap(0.15 + (i % 8) * 0.85 / 8)  # Avoid too light colors by starting at 0.15
            colors.append(color)
        
        plt.figure(figsize=(12, 8))
        
        # Plot each agent's progression with its unique color
        for agent_name, color in zip(agent_names, colors):
            agent_data = df[df['agent_name'] == agent_name]
            agent_type = self.agent_types[agent_name]
            
            # Calculate cumulative win rate
            wins = agent_data.groupby('game_id')['won'].mean()
            cumulative_wins = wins.expanding().mean() * 100
            
            plt.plot(cumulative_wins.index, cumulative_wins.values,
                    label=f'{agent_name} ({agent_type})',
                    color=color,
                    linewidth=2, marker='o', markersize=4)
        
        plt.title('Individual Agent Win Rate Over Time (Cumulative)', 
                 pad=20, fontsize=12, fontweight='bold')
        plt.xlabel('Game Number')
        plt.ylabel('Win Rate (%)')
        
        # Adjust legend for better readability with more agents
        plt.legend(title='Agents', 
                  bbox_to_anchor=(1.05, 1), 
                  loc='upper left',
                  borderaxespad=0,
                  fontsize=8,
                  ncol=max(1, n_agents // 8))  # Use multiple columns if many agents
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        suffix = f"_{plt_suffix}" if plt_suffix else ""
        plt.savefig(f'{output_dir}/individual_performance_over_time{suffix}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

    def plot_agent_win_progression(self, output_dir: str = 'reports', plt_suffix: str = ''):
        pass

    def plot_bid_distribution(self, output_dir: str, plt_suffix: str):
        """Plot distribution of bid quantities to analyze bidding patterns."""
        df = self.to_dataframe()
        plt.figure(figsize=(10, 6))
        
        for agent_type, agent_data in df.groupby('agent_type'):
            try:
                if len(agent_data['avg_bid_quantity'].unique()) > 1:
                    sns.kdeplot(data=agent_data['avg_bid_quantity'], label=agent_type)
                else:
                    # If all values are the same, plot a single vertical line
                    value = agent_data['avg_bid_quantity'].iloc[0]
                    plt.axvline(x=value, label=f"{agent_type} (constant={value:.2f})", 
                              linestyle='--', alpha=0.7)
            except Exception as e:
                if self.verbose >= 1:
                    print(f"Warning: Could not plot distribution for {agent_type}: {str(e)}")
                continue
        
        plt.title('Bid Quantity Distribution by Agent Type', pad=20, fontsize=12, fontweight='bold')
        plt.xlabel('Average Bid Quantity')
        plt.ylabel('Density')
        plt.legend(title='Agent Type')
        
        suffix = f"_{plt_suffix}" if plt_suffix else ""
        plt.savefig(f'{output_dir}/bid_distribution{suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_bluff_timing(self, output_dir: str, plt_suffix: str):
        """Plot the average quantity of dice that was bid in the previous round when agents call bluff.
        This helps us understand what bid quantities trigger different agents to call bluff."""
        df = self.to_dataframe()
        
        if df.empty:
            if self.verbose >= 1:
                print("Warning: No data available for bluff timing plot")
            return
            
        plt.figure(figsize=(12, 8))  # Made plot larger
        
        # Explode list columns to get individual values
        success_data = df.explode('prev_bid_quantity_successful_bluff')
        failed_data = df.explode('prev_bid_quantity_failed_bluff')
        
        if self.verbose >= 1:
            print("\nDEBUG: Bluff Timing Data")
            print("\nSuccessful bluff calls:")
            for agent_type in df['agent_type'].unique():
                data = success_data[success_data['agent_type'] == agent_type]['prev_bid_quantity_successful_bluff']
                print(f"{agent_type}: {list(data)}")
            print("\nFailed bluff calls:")
            for agent_type in df['agent_type'].unique():
                data = failed_data[failed_data['agent_type'] == agent_type]['prev_bid_quantity_failed_bluff']
                print(f"{agent_type}: {list(data)}")
        
        # Convert to numeric, dropping any non-numeric values
        success_data['prev_bid_quantity_successful_bluff'] = pd.to_numeric(
            success_data['prev_bid_quantity_successful_bluff'], 
            errors='coerce'
        )
        failed_data['prev_bid_quantity_failed_bluff'] = pd.to_numeric(
            failed_data['prev_bid_quantity_failed_bluff'], 
            errors='coerce'
        )
        
        # Drop NaN values
        success_data = success_data.dropna(subset=['prev_bid_quantity_successful_bluff'])
        failed_data = failed_data.dropna(subset=['prev_bid_quantity_failed_bluff'])
        
        if success_data.empty and failed_data.empty:
            if self.verbose >= 1:
                print("Warning: No valid bluff timing data after processing")
            return
        
        # Calculate means for each agent type
        success_means = success_data.groupby('agent_type')['prev_bid_quantity_successful_bluff'].mean()
        failed_means = failed_data.groupby('agent_type')['prev_bid_quantity_failed_bluff'].mean()
        
        if self.verbose >= 1:
            print("\nBluff Timing Statistics:")
            print("\nSuccessful Bluffs:")
            print(success_means)
            print("\nFailed Bluffs:")
            print(failed_means)
        
        # Get unique agent types and sort them
        agent_types = sorted(df['agent_type'].unique())
        x = np.arange(len(agent_types))
        width = 0.35
        
        # Create bars with error bars
        success_std = success_data.groupby('agent_type')['prev_bid_quantity_successful_bluff'].std()
        failed_std = failed_data.groupby('agent_type')['prev_bid_quantity_failed_bluff'].std()
        
        plt.bar(x - width/2, 
               [success_means.get(agent, 0) for agent in agent_types], 
               width, 
               yerr=[success_std.get(agent, 0) for agent in agent_types],
               label='Successful Bluff Calls',
               capsize=5)
        plt.bar(x + width/2, 
               [failed_means.get(agent, 0) for agent in agent_types], 
               width,
               yerr=[failed_std.get(agent, 0) for agent in agent_types],
               label='Failed Bluff Calls',
               capsize=5)
        
        plt.title('Previous Bid Quantities that Trigger Bluff Calls', pad=20, fontsize=12, fontweight='bold')
        plt.xlabel('Agent Type')
        plt.ylabel('Average Previous Bid Quantity')
        plt.xticks(x, agent_types)
        plt.legend()
        
        # Add value labels on bars with more padding
        for i, agent in enumerate(agent_types):
            success_val = success_means.get(agent, 0)
            failed_val = failed_means.get(agent, 0)
            if success_val > 0:
                plt.text(i - width/2 + 0.1, success_val + 0.2, f'{success_val:.1f}', 
                        ha='center', va='bottom')
            if failed_val > 0:
                plt.text(i + width/2 + 0.1, failed_val + 0.2, f'{failed_val:.1f}', 
                        ha='center', va='bottom')
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        suffix = f"_{plt_suffix}" if plt_suffix else ""
        plt.savefig(f'{output_dir}/bluff_timing{suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_probability_accuracy(self, output_dir: str, plt_suffix: str):
        """Plot predicted probability vs actual outcomes for informed agents."""
        df = self.to_dataframe()
        
        if df.empty:
            if self.verbose >= 1:
                print("Warning: No data available for probability accuracy plot")
            return
        
        # Filter for Informed agents (note: agent type is stored without 'Agent' suffix)
        df_exploded = df[df['agent_type'] == 'Informed'].explode(['predicted_probability', 'actual_outcome'])
        
        # Convert to numeric, dropping any non-numeric values
        df_exploded['predicted_probability'] = pd.to_numeric(df_exploded['predicted_probability'], errors='coerce')
        df_exploded['actual_outcome'] = pd.to_numeric(df_exploded['actual_outcome'], errors='coerce')
        
        # Drop NaN values
        df_exploded = df_exploded.dropna(subset=['predicted_probability', 'actual_outcome'])
        
        if df_exploded.empty:
            if self.verbose >= 1:
                print("Warning: No valid probability accuracy data for Informed agents")
            return
        
        plt.figure(figsize=(10, 8))
        
        # Create hexbin plot instead of scatter for dense data
        hb = plt.hexbin(df_exploded['predicted_probability'], 
                       df_exploded['actual_outcome'],
                       gridsize=20,  # Adjust number of hexagons
                       cmap='YlOrRd',  # Use a sequential colormap
                       mincnt=1,  # Minimum points for a hex to be colored
                       bins='log')  # Use log scale for better color distribution
        
        # Add colorbar to show density
        cb = plt.colorbar(hb, label='Log10(Count)')
        
        # Add perfect prediction line
        plt.plot([0, 1], [0, 1], 'b--', label='Perfect Prediction', linewidth=2)
        
        # Add trend line
        z = np.polyfit(df_exploded['predicted_probability'], 
                      df_exploded['actual_outcome'], 1)
        p = np.poly1d(z)
        plt.plot(df_exploded['predicted_probability'], 
                p(df_exploded['predicted_probability']),
                "g-", alpha=0.8, linewidth=2,
                label=f'Trend Line (y={z[0]:.2f}x+{z[1]:.2f})')
        
        # Calculate and display correlation
        corr = df_exploded['predicted_probability'].corr(df_exploded['actual_outcome'])
        plt.text(0.02, 0.95, 
                f'Prediction-Reality Correlation: {corr:.3f}\n'
                f'(1 = perfect, 0 = none, -1 = inverse)',
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Calculate and display additional statistics
        mse = ((df_exploded['predicted_probability'] - df_exploded['actual_outcome']) ** 2).mean()
        plt.text(0.02, 0.85,
                f'Mean Squared Error: {mse:.3f}\n'
                f'Root MSE: {np.sqrt(mse):.3f}',
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.title('Probability Prediction Accuracy (Hexbin Density Plot)')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Actual Outcome')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower right')
        plt.tight_layout()
        
        # Set equal aspect ratio for better visualization
        plt.gca().set_aspect('equal', adjustable='box')
        
        suffix = f"_{plt_suffix}" if plt_suffix else ""
        plt.savefig(f'{output_dir}/probability_accuracy{suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_adaptive_learning(self, output_dir: str, plt_suffix: str):
        """Plot adaptive agent's learning progression."""
        df = self.to_dataframe()
        
        if df.empty:
            if self.verbose >= 1:
                print("Warning: No data available for adaptive learning plot")
            return
        
        # Filter for Adaptive agents (note: agent type is stored without 'Agent' suffix)
        df_adaptive = df[df['agent_type'] == 'Adaptive'].copy()
        
        if self.verbose >= 1:
            print("\nDebugging Adaptive Learning Plot:")
            print(f"Total agents in data: {len(df['agent_type'].unique())}")
            print("Agent types present:", df['agent_type'].unique())
            print(f"Number of adaptive agents found: {len(df_adaptive)}")
            print("\nSample of adaptive agent data:")
            if not df_adaptive.empty:
                print("Predicted bluff rates:", df_adaptive['predicted_bluff_rate'].iloc[0][:5])
                print("Actual bluff rates:", df_adaptive['actual_bluff_rate'].iloc[0][:5])
                print("Bluff thresholds:", df_adaptive['bluff_threshold'].iloc[0][:5])
            if len(df_adaptive) == 0:
                print("Warning: No adaptive agents found in the data")
                print("This might happen if:")
                print("1. No adaptive agents were included in the simulation")
                print("2. The agent type name doesn't match 'Adaptive'")
                print("3. The adaptive agent's metrics weren't properly recorded")
                return
        
        # Create lists to store exploded data
        game_ids = []
        agent_names = []
        predicted_rates = []
        actual_rates = []
        thresholds = []
        
        # Manually explode the data
        for idx, row in df_adaptive.iterrows():
            pred_rate = row['predicted_bluff_rate']
            act_rate = row['actual_bluff_rate']
            thresh = row['bluff_threshold']
            
            # Get the minimum length to ensure alignment
            min_len = min(len(pred_rate), len(act_rate), len(thresh))
            
            # Add data points
            for i in range(min_len):
                game_ids.append(row['game_id'])
                agent_names.append(row['agent_name'])
                predicted_rates.append(pred_rate[i])
                actual_rates.append(act_rate[i])
                thresholds.append(thresh[i])
        
        # Create new DataFrame with exploded data
        df_exploded = pd.DataFrame({
            'game_id': game_ids,
            'agent_name': agent_names,
            'predicted_bluff_rate': predicted_rates,
            'actual_bluff_rate': actual_rates,
            'bluff_threshold': thresholds
        })
        
        # Convert to numeric, dropping any non-numeric values
        for col in ['predicted_bluff_rate', 'actual_bluff_rate', 'bluff_threshold']:
            df_exploded[col] = pd.to_numeric(df_exploded[col], errors='coerce')
        
        # Drop NaN values
        df_exploded = df_exploded.dropna(subset=['predicted_bluff_rate', 'actual_bluff_rate', 'bluff_threshold'])
        
        if df_exploded.empty:
            if self.verbose >= 1:
                print("Warning: No valid adaptive learning data")
            return
            
        if self.verbose >= 2:
            print("\nAdaptive Learning Data:")
            print(f"Number of data points: {len(df_exploded)}")
            print("\nSample of values:")
            print(df_exploded[['predicted_bluff_rate', 'actual_bluff_rate', 'bluff_threshold']].head())
        
        # Create figure with two subplots with increased height and spacing
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14), height_ratios=[1, 1])
        
        # Add super title with more padding
        fig.suptitle('Adaptive Agent Learning Analysis', 
                    fontsize=14, 
                    fontweight='bold', 
                    y=0.98)  # Move title up
        
        # Add more space between subplots
        plt.subplots_adjust(hspace=0.3)
        
        # Plot 1: Learning Progression Over Time
        # First sort by game_id and create move numbers
        df_exploded = df_exploded.sort_values('game_id')
        df_exploded['move_number'] = df_exploded.groupby('game_id').cumcount() + 1
        df_exploded = df_exploded.sort_values(['game_id', 'move_number'])
        df_exploded['global_move_number'] = range(1, len(df_exploded) + 1)
        
        # Create bins of 50 moves
        bin_size = 50  # Increased from 20 to 50 for smoother visualization
        df_exploded['move_bin'] = df_exploded['global_move_number'].apply(lambda x: ((x-1) // bin_size) * bin_size + bin_size/2)
        
        # Calculate statistics for each bin
        binned_stats = df_exploded.groupby('move_bin').agg({
            'predicted_bluff_rate': ['mean', 'std'],
            'actual_bluff_rate': ['mean', 'std'],
            'bluff_threshold': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        binned_stats.columns = ['move_bin', 
                              'predicted_mean', 'predicted_std',
                              'actual_mean', 'actual_std',
                              'threshold_mean', 'threshold_std']
        
        # Plot main lines using binned means
        ax1.plot(binned_stats['move_bin'], binned_stats['predicted_mean'],
                label='Predicted Bluff Rate', color='blue', linewidth=2, alpha=0.7)
        ax1.plot(binned_stats['move_bin'], binned_stats['actual_mean'],
                label='Actual Bluff Rate', color='orange', linewidth=2, alpha=0.7)
        ax1.plot(binned_stats['move_bin'], binned_stats['threshold_mean'],
                label='Bluff Threshold', color='green', linewidth=2, alpha=0.7)
        
        # Add confidence intervals
        for col_mean, col_std, color in [
            ('predicted_mean', 'predicted_std', 'blue'),
            ('actual_mean', 'actual_std', 'orange'),
            ('threshold_mean', 'threshold_std', 'green')
        ]:
            ax1.fill_between(binned_stats['move_bin'],
                           binned_stats[col_mean] - binned_stats[col_std],
                           binned_stats[col_mean] + binned_stats[col_std],
                           alpha=0.1, color=color)
        
        # Add light scatter points for raw values with lighter gray color
        ax1.scatter(df_exploded['global_move_number'], df_exploded['predicted_bluff_rate'],
                   alpha=0.05, color='lightgray', s=2)  # Reduced alpha and size, lighter color
        ax1.scatter(df_exploded['global_move_number'], df_exploded['actual_bluff_rate'],
                   alpha=0.05, color='lightgray', s=2)
        ax1.scatter(df_exploded['global_move_number'], df_exploded['bluff_threshold'],
                   alpha=0.05, color='lightgray', s=2)
        
        # Add vertical lines at game boundaries with lighter color
        game_boundaries = df_exploded.groupby('game_id')['global_move_number'].last()
        for game_end in game_boundaries:
            ax1.axvline(x=game_end, color='lightgray', linestyle='--', alpha=0.1)  # Lighter color and reduced alpha
        
        ax1.set_title('Learning Progression Over Time')
        ax1.set_xlabel('Move Number (Across All Games)')
        ax1.set_ylabel('Rate')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Calculate and display convergence metrics
        final_error = abs(df_exploded['predicted_bluff_rate'] - df_exploded['actual_bluff_rate']).iloc[-10:].mean()
        ax1.text(0.02, 0.95, 
                f'Recent Prediction Accuracy:\n'
                f'Mean Error (last 10 moves): {final_error:.3f}\n'
                f'(0 = perfect, 1 = worst)',
                transform=ax1.transAxes,
                bbox=dict(facecolor='white', alpha=0.8))
        
        # Plot 2: Prediction Accuracy
        # Create hexbin plot instead of scatter for dense data
        hb = ax2.hexbin(df_exploded['predicted_bluff_rate'], 
                       df_exploded['actual_bluff_rate'],
                       gridsize=20,  # Adjust number of hexagons
                       cmap='YlOrRd',  # Use a sequential colormap
                       mincnt=1,  # Minimum points for a hex to be colored
                       bins='log')  # Use log scale for better color distribution
        
        # Add colorbar to show density
        plt.colorbar(hb, ax=ax2, label='Log10(Count)')
        
        # Add perfect prediction line
        ax2.plot([0, 1], [0, 1], 'b--', label='Perfect Prediction', linewidth=2)
        
        # Add trend line if there's enough variance
        pred_std = df_exploded['predicted_bluff_rate'].std()
        actual_std = df_exploded['actual_bluff_rate'].std()
        
        if pred_std > 0 and actual_std > 0:
            z = np.polyfit(df_exploded['predicted_bluff_rate'], 
                         df_exploded['actual_bluff_rate'], 1)
            p = np.poly1d(z)
            ax2.plot(df_exploded['predicted_bluff_rate'], 
                    p(df_exploded['predicted_bluff_rate']),
                    "g-", alpha=0.8, linewidth=2,
                    label=f'Trend Line (y={z[0]:.2f}x+{z[1]:.2f})')
            
            # Calculate and display correlation
            corr = df_exploded['predicted_bluff_rate'].corr(df_exploded['actual_bluff_rate'])
            ax2.text(0.02, 0.95, 
                    f'Prediction-Reality Correlation: {corr:.3f}\n'
                    f'(1 = perfect, 0 = none, -1 = inverse)',
                    transform=ax2.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8))
        else:
            ax2.text(0.02, 0.95,
                    'Insufficient variance for correlation',
                    transform=ax2.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8))
        
        ax2.set_title('Prediction Accuracy')
        ax2.set_xlabel('Predicted Bluff Rate')
        ax2.set_ylabel('Actual Bluff Rate')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        suffix = f"_{plt_suffix}" if plt_suffix else ""
        plt.savefig(f'{output_dir}/adaptive_learning{suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_llm_reasoning(self, output_dir: str, plt_suffix: str):
        """Analyze and plot LLM agent's reasoning patterns using clustering analysis."""
        from reasoning_analysis import ReasoningAnalyzer
        import plotly.io as pio
        
        df = self.to_dataframe()
        
        if df.empty:
            if self.verbose >= 1:
                print("Warning: No data available for LLM reasoning plot")
            return
        
        # Filter for LLM agents and explode decision data
        df_exploded = df[df['agent_type'] == 'LLM'].explode(['decision_type', 'successful_move'])
        
        if df_exploded.empty:
            if self.verbose >= 1:
                print("Warning: No valid LLM reasoning data")
            return
            
        # Convert boolean successful_move to numeric
        df_exploded['successful_move'] = df_exploded['successful_move'].map({True: 1, False: 0})
        
        # Initialize ReasoningAnalyzer with config
        config = {
            'embedding_model': 'all-MiniLM-L6-v2',
            'clustering': {
                'min_cluster_size': 5,
                'min_samples': 3
            },
            'visualization': {
                'output_formats': ['html', 'png'],
                'interactive': True
            }
        }
        analyzer = ReasoningAnalyzer(config)
        
        # Generate embeddings and cluster the decision types
        decision_types = df_exploded['decision_type'].tolist()
        embeddings = analyzer.generate_embeddings(decision_types)
        similarities = analyzer.calculate_similarity(embeddings)
        clusters = analyzer.cluster_reasonings(similarities)
        cluster_labels = clusters['labels']
        
        # Get cluster summaries
        summaries = analyzer.summarize_clusters(clusters, decision_types)
        
        # Create cluster labels with categories
        cluster_labels_with_category = {}
        cluster_labels_simple = {}
        cluster_labels_detailed = {}
        
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:
                label = "Mixed Strategy"
                cluster_labels_with_category[cluster_id] = label
                cluster_labels_simple[cluster_id] = "Mixed"
                cluster_labels_detailed[cluster_id] = label
            else:
                summary = summaries.get(cluster_id)
                simple_label = f"Cluster {cluster_id}"
                cluster_labels_simple[cluster_id] = simple_label
                if summary:
                    detailed_label = f"{simple_label}: {summary.category_name}"
                    cluster_labels_with_category[cluster_id] = detailed_label
                    cluster_labels_detailed[cluster_id] = detailed_label
                else:
                    cluster_labels_with_category[cluster_id] = simple_label
                    cluster_labels_detailed[cluster_id] = simple_label
        
        # Add cluster labels to dataframe
        df_exploded['cluster'] = cluster_labels
        
        # Function to generate distinct colors
        def generate_colors(n):
            """Generate n distinct colors using HSV color space"""
            colors = []
            for i in range(n):
                hue = i / n
                # Use high saturation and value for vivid colors
                saturation = 0.7
                value = 0.9
                rgb = colorsys.hsv_to_rgb(hue, saturation, value)
                # Convert to hex
                hex_color = '#{:02x}{:02x}{:02x}'.format(
                    int(rgb[0] * 255),
                    int(rgb[1] * 255),
                    int(rgb[2] * 255)
                )
                colors.append(hex_color)
            return colors
        
        # Get unique clusters (excluding -1) and generate colors
        unique_clusters = sorted(list(set(cluster_labels)))
        n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)  # Exclude noise cluster
        distinct_colors = generate_colors(n_clusters)
        
        # Assign colors to remaining clusters
        cluster_colors = {}
        if -1 in unique_clusters:
            cluster_colors[-1] = '#808080'  # Gray for noise/uncategorized
            unique_clusters.remove(-1)
        
        # Assign colors to remaining clusters
        for cluster, color in zip(unique_clusters, distinct_colors):
            cluster_colors[cluster] = color
        
        # Create figure with 2x2 subplots for metrics analysis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Distribution of Decision Strategies',
                'Success Rate by Strategy',
                'Strategy Evolution by Game',
                'Strategy Success Rate by Game'
            ),
            specs=[
                [{"type": "pie"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.12  # Increased spacing between columns
        )
        
        # 1. Distribution of Clusters (Pie Chart)
        cluster_counts = df_exploded['cluster'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=[cluster_labels_simple[c] for c in cluster_counts.index],
                values=cluster_counts.values,
                hole=0.3,
                textinfo='percent+label',
                marker=dict(colors=[cluster_colors[c] for c in cluster_counts.index]),
                name='Distribution',
                legendgroup='distribution',
                legendgrouptitle_text="Distribution of Strategies"
            ),
            row=1, col=1
        )
        
        # 2. Success Rate by Cluster (Bar Chart)
        success_by_cluster = df_exploded.groupby('cluster')['successful_move'].agg(['mean', 'count'])
        fig.add_trace(
            go.Bar(
                x=[cluster_labels_simple[c] for c in success_by_cluster.index],
                y=success_by_cluster['mean'],
                text=[f"{count} decisions<br>{rate:.1%} success rate" 
                     for rate, count in zip(success_by_cluster['mean'], success_by_cluster['count'])],
                textposition='auto',
                marker_color=[cluster_colors[c] for c in success_by_cluster.index],
                name='Success Rate',
                legendgroup='success_rate',
                legendgrouptitle_text="Success Rate by Strategy",
                showlegend=False  # Hide this since it's just one bar series
            ),
            row=1, col=2
        )
        
        # 3. Strategy Evolution by Game
        strategy_evolution = df_exploded.groupby(['game_id', 'cluster']).size().unstack(fill_value=0)
        strategy_evolution_pct = strategy_evolution.div(strategy_evolution.sum(axis=1), axis=0)
        
        for cluster in strategy_evolution_pct.columns:
            fig.add_trace(
                go.Scatter(
                    x=strategy_evolution_pct.index,
                    y=strategy_evolution_pct[cluster] * 100,
                    name=cluster_labels_detailed[cluster],  # Use detailed label here
                    mode='lines+markers',
                    line=dict(color=cluster_colors[cluster]),
                    hovertemplate=f"Game %{{x}}<br>%{{y:.1f}}% of decisions",
                    legendgroup='evolution',
                    legendgrouptitle_text="Strategy Evolution"
                ),
                row=2, col=1
            )
        
        # 4. Success Rate by Game and Strategy
        success_by_game_strategy = df_exploded.groupby(['game_id', 'cluster'])['successful_move'].mean().unstack()
        
        for cluster in success_by_game_strategy.columns:
            fig.add_trace(
                go.Scatter(
                    x=success_by_game_strategy.index,
                    y=success_by_game_strategy[cluster] * 100,
                    name=cluster_labels_detailed[cluster],  # Use detailed label here
                    mode='lines+markers',
                    line=dict(color=cluster_colors[cluster]),
                    hovertemplate=f"Game %{{x}}<br>%{{y:.1f}}% success rate",
                    legendgroup='success_evolution',
                    legendgrouptitle_text="Success Rate Evolution"
                ),
                row=2, col=2
            )
        
        # Update layout with better legend placement and sizing
        fig.update_layout(
            height=1000,
            width=1800,  # Increased width
            showlegend=True,
            title_text="LLM Agent Decision Strategy Analysis",
            title_x=0.5,
            title_font_size=20,
            legend=dict(
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=1.02,
                orientation="v",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1,
                tracegroupgap=30
            ),
            # Add margins to ensure nothing gets cut off
            margin=dict(
                l=50,   # left margin
                r=250,  # right margin for legend
                t=100,  # top margin
                b=50    # bottom margin
            )
        )
        
        # Update axes labels and formatting
        fig.update_yaxes(title_text="Percentage of Decisions", row=2, col=1)
        fig.update_yaxes(title_text="Success Rate (%)", row=2, col=2)
        fig.update_xaxes(title_text="Game Number", row=2, col=1)
        fig.update_xaxes(title_text="Game Number", row=2, col=2)
        
        # Update subplot titles to be more descriptive
        for i in range(len(fig.layout.annotations)):
            fig.layout.annotations[i].font.size = 14
        
        # Update plot labels to use simple labels for axes and detailed labels for legend/hover
        fig.update_layout(
            showlegend=True,
            legend=dict(
                title="Strategy Clusters",
                orientation="v",
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=1.02
            )
        )

        # Update traces to use detailed labels in legend and hover, but simple labels on axes
        for trace in fig.data:
            if hasattr(trace, 'name') and trace.name in cluster_labels_with_category.values():
                # Convert from detailed back to cluster ID
                cluster_id = next(k for k, v in cluster_labels_with_category.items() if v == trace.name)
                # Update to use detailed label in legend
                trace.name = cluster_labels_detailed[cluster_id]
                # Update hover template to show detailed information
                if hasattr(trace, 'hovertemplate'):
                    # Replace the current label with the simple label
                    hover_text = trace.hovertemplate.replace(
                        cluster_labels_with_category[cluster_id],
                        cluster_labels_simple[cluster_id]
                    )
                    # Add strategy information to hover
                    trace.hovertemplate = hover_text + f"<br>Strategy: {cluster_labels_detailed[cluster_id]}<extra></extra>"
                    # Get the length of data from x or y values
                    data_length = len(trace.x) if hasattr(trace, 'x') else len(trace.y)
                    trace.customdata = [[cluster_labels_detailed[cluster_id]] for _ in range(data_length)]
        
        # Save metrics analysis plots
        suffix = f"_{plt_suffix}" if plt_suffix else ""
        pio.write_html(fig, f'{output_dir}/llm_reasoning_metrics{suffix}.html')
        pio.write_image(fig, f'{output_dir}/llm_reasoning_metrics{suffix}.png')
        
        # Generate ReasoningAnalyzer visualizations
        analyzer.render_visualizations(
            summaries=summaries,
            text_data=decision_types,
            output_dir=output_dir,
            filename_suffix=suffix
        )
        
    def generate_reports(self, output_dir: str = 'reports', plt_suffix: str = '', from_file: str = ''):
        """Generate comprehensive performance reports"""
        import os
        
        # Create base output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # If plt_suffix is provided, create a subdirectory for this run
        if plt_suffix:
            output_dir = os.path.join(output_dir, plt_suffix.strip('_'))
            os.makedirs(output_dir, exist_ok=True)
        
        if from_file:
            df = pd.read_csv(from_file)
        else:
            df = self.to_dataframe()
            # Save the raw data to csv
            df.to_csv(os.path.join(f'{output_dir}', 'raw_data.csv'), index=False)


        agent_types = sorted(df['agent_type'].unique())
        
        # Generate all plots
        self.plot_win_rates(output_dir, '')
        self.plot_individual_win_rates(output_dir, '')  # Add the new plot
        self.plot_survival_time(output_dir, '')
        self.plot_bluffing_behavior(output_dir, '')
        self.plot_bid_patterns(output_dir, '')
        self.plot_win_progression(output_dir, '')  # Aggregate by type
        self.plot_individual_win_progression(output_dir, '')  # Individual agents
        self.plot_elimination_distribution(output_dir, '')
        self.plot_individual_agent_metrics(output_dir, '')
        
        # New visualization methods
        self.plot_bid_distribution(output_dir, '')
        self.plot_bluff_timing(output_dir, '')
        self.plot_probability_accuracy(output_dir, '')
        self.plot_adaptive_learning(output_dir, '')
        self.plot_llm_reasoning(output_dir, '')
        
        # Generate summary statistics
        summary = {
            'Overall Statistics': df.groupby('agent_type').agg({
                'won': ['mean', 'count'],
                'survival_time': ['mean', 'std'],
                'bluff_rate': 'mean',
                'successful_bluffs': 'mean',
                'successful_catches': 'mean',
                'final_lives': 'mean'
            }).round(3)
        }
        
        # Reorder summary statistics
        summary['Overall Statistics'] = summary['Overall Statistics'].reindex(agent_types)
        
        # Save summary to CSV
        for name, stats in summary.items():
            stats.to_csv(f'{output_dir}/{name.lower().replace(" ", "_")}.csv')

    def plot_individual_win_rates(self, output_dir: str, plt_suffix: str):
        """Plot win rates for each individual agent."""
        df = self.to_dataframe()
        
        # Get total games count
        total_games = len(self.games)
        
        # Calculate wins per agent
        wins_per_agent = df[df['won'] == 1].groupby(['agent_name', 'agent_type']).size()
        
        # Calculate win rate as percentage
        win_rates = (wins_per_agent / total_games * 100).fillna(0)
        
        # Create a DataFrame with agent names and their types
        win_rates_df = pd.DataFrame(win_rates).reset_index()
        win_rates_df.columns = ['agent_name', 'agent_type', 'win_rate']
        
        # Sort by agent type and win rate
        win_rates_df = win_rates_df.sort_values(['agent_type', 'win_rate'], ascending=[True, False])
        
        # Plot
        plt.figure(figsize=(12, 6))
        colors = self.get_plot_colors()
        
        # Create color mapping for agent types
        agent_types = sorted(df['agent_type'].unique())
        color_dict = {agent_type: colors[i] for i, agent_type in enumerate(agent_types)}
        bar_colors = [color_dict[agent_type] for agent_type in win_rates_df['agent_type']]
        
        ax = plt.gca()
        bars = ax.bar(range(len(win_rates_df)), win_rates_df['win_rate'], color=bar_colors)
        
        # Customize the plot
        plt.title('Individual Agent Win Rates', pad=20, fontsize=12, fontweight='bold')
        plt.ylabel('Win Rate (%)')
        plt.xlabel('Agent')
        
        # Set x-axis labels
        plt.xticks(range(len(win_rates_df)), 
                  win_rates_df['agent_name'],  # Just use agent names without the type
                  rotation=45, ha='right')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2 + 0.1, height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Add legend for agent types
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color_dict[agent_type], 
                                       label=agent_type) for agent_type in agent_types]
        plt.legend(handles=legend_elements, title='Agent Types', 
                  bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        suffix = f"_{plt_suffix}" if plt_suffix else ""
        plt.savefig(f'{output_dir}/individual_win_rates{suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()
