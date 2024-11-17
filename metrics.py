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
    dice_remaining_successful_bluff: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))
    dice_remaining_failed_bluff: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))
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
                'dice_remaining_successful_bluff': game.dice_remaining_successful_bluff.get(game.winner, []),
                'dice_remaining_failed_bluff': game.dice_remaining_failed_bluff.get(game.winner, []),
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
                        'dice_remaining_successful_bluff': game.dice_remaining_successful_bluff.get(agent_name, []),
                        'dice_remaining_failed_bluff': game.dice_remaining_failed_bluff.get(agent_name, []),
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
        #     list_cols = ['dice_remaining_successful_bluff', 'dice_remaining_failed_bluff',
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
        agent_types = sorted(df['agent_type'].unique())
        colors = self.get_plot_colors()
        suffix = f"_{plt_suffix}" if plt_suffix else ""
        
        plt.figure(figsize=(10, 6))
        win_rates = df.groupby('agent_type')['won'].mean() * 100  # Convert to percentage
        win_rates = win_rates.reindex(agent_types)
        ax = win_rates.plot(kind='bar', color=[colors[i] for i in range(len(agent_types))])
        plt.title('Win Rates by Agent Type', pad=20, fontsize=12, fontweight='bold')
        plt.ylabel('Win Rate (%)')
        
        # Add value labels on bars
        for i, v in enumerate(win_rates):
            ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/win_rates{suffix}.png', dpi=300, bbox_inches='tight')
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
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        metrics_df = df.groupby('agent_type').agg({
            'bluff_rate': 'mean',
            'successful_bluffs': 'mean',
            'successful_catches': 'mean'
        })
        metrics_df = metrics_df.reindex(agent_types)
        
        bar_width = 0.25
        x = np.arange(len(agent_types))
        
        ax.bar(x - bar_width, metrics_df['bluff_rate'], bar_width, 
               label='Bluff Rate', color=colors[0])
        ax.bar(x, metrics_df['successful_bluffs'], bar_width,
               label='Successful Bluffs per Game', color=colors[1])
        ax.bar(x + bar_width, metrics_df['successful_catches'], bar_width,
               label='Successful Catches per Game', color=colors[2])
        
        ax.set_title('Bluffing Behavior by Agent Type', pad=20, fontsize=12, fontweight='bold')
        ax.set_ylabel('Value')
        ax.set_xticks(x)
        ax.set_xticklabels(agent_types)
        
        for i in range(len(metrics_df.index)):
            bluff_rate = metrics_df['bluff_rate'].iloc[i]
            ax.text(i - bar_width, bluff_rate + 0.01, f'{bluff_rate*100:.1f}%', 
                   ha='center', va='bottom', fontweight='bold')
            
            sbluffs = metrics_df['successful_bluffs'].iloc[i]
            ax.text(i, sbluffs + 0.01, f'{sbluffs:.1f}', 
                   ha='center', va='bottom', fontweight='bold')
            
            scatches = metrics_df['successful_catches'].iloc[i]
            ax.text(i + bar_width, scatches + 0.01, f'{scatches:.1f}', 
                   ha='center', va='bottom', fontweight='bold')
        
        ax.legend(bbox_to_anchor=(0.98, 0.98), loc='upper right',
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
        
        # Add value labels on bars
        for i, v in enumerate(bid_means):
            plt.text(i, v + 0.1, f'{v:.2f}', ha='center')
        
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
        metrics_df = df.groupby('agent_name').agg({
            'won': 'sum',  # Total wins
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
        labels = ['Total Wins', 'Successful Bluffs/Game', 'Failed Bluffs/Game', 'Successful Catches/Game']
        
        for i, (metric, label) in enumerate(zip(metrics, labels)):
            offset = (i - 1.5) * bar_width
            bars = ax.bar(x + offset, metrics_df[metric], bar_width,
                         label=label, color=plt.cm.Set2(i/4))
            
            # Add value labels on bars
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar_width/2, height + 0.01,
                       f'{height:.1f}', ha='center', va='bottom',
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
        
        for agent_type in sorted(df['agent_type'].unique()):
            agent_data = df[df['agent_type'] == agent_type]
            sns.kdeplot(data=agent_data['avg_bid_quantity'], label=agent_type)
        
        plt.title('Bid Quantity Distribution by Agent Type', pad=20, fontsize=12, fontweight='bold')
        plt.xlabel('Average Bid Quantity')
        plt.ylabel('Density')
        plt.legend(title='Agent Type')
        
        suffix = f"_{plt_suffix}" if plt_suffix else ""
        plt.savefig(f'{output_dir}/bid_distribution{suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_bluff_timing(self, output_dir: str, plt_suffix: str):
        """Plot average number of dice remaining when bluff is called."""
        df = self.to_dataframe()
        
        if df.empty:
            if self.verbose >= 1:
                print("Warning: No data available for bluff timing plot")
            return
            
        plt.figure(figsize=(10, 6))
        
        # Explode list columns to get individual values
        success_data = df.explode('dice_remaining_successful_bluff')
        failed_data = df.explode('dice_remaining_failed_bluff')
        
        # Convert to numeric, dropping any non-numeric values
        success_data['dice_remaining_successful_bluff'] = pd.to_numeric(
            success_data['dice_remaining_successful_bluff'], 
            errors='coerce'
        )
        failed_data['dice_remaining_failed_bluff'] = pd.to_numeric(
            failed_data['dice_remaining_failed_bluff'], 
            errors='coerce'
        )
        
        # Drop NaN values
        success_data = success_data.dropna(subset=['dice_remaining_successful_bluff'])
        failed_data = failed_data.dropna(subset=['dice_remaining_failed_bluff'])
        
        if success_data.empty and failed_data.empty:
            if self.verbose >= 1:
                print("Warning: No valid bluff timing data after processing")
            return
        
        # Calculate means for each agent type
        success_means = success_data.groupby('agent_type')['dice_remaining_successful_bluff'].mean()
        failed_means = failed_data.groupby('agent_type')['dice_remaining_failed_bluff'].mean()
        
        # Get unique agent types and sort them
        agent_types = sorted(df['agent_type'].unique())
        x = np.arange(len(agent_types))
        width = 0.35
        
        # Create bars with error bars
        success_std = success_data.groupby('agent_type')['dice_remaining_successful_bluff'].std()
        failed_std = failed_data.groupby('agent_type')['dice_remaining_failed_bluff'].std()
        
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
        
        plt.title('Average Dice Remaining When Calling Bluff', pad=20, fontsize=12, fontweight='bold')
        plt.xlabel('Agent Type')
        plt.ylabel('Average Dice Remaining')
        plt.xticks(x, agent_types)
        plt.legend()
        
        # Add value labels on bars
        def add_value_labels(x, values, offset):
            for i, v in enumerate(values):
                if not np.isnan(v):
                    plt.text(x[i] + offset, v + 0.1, f'{v:.1f}', 
                           ha='center', fontsize=8)
        
        add_value_labels(x, [success_means.get(agent, 0) for agent in agent_types], -width/2)
        add_value_labels(x, [failed_means.get(agent, 0) for agent in agent_types], width/2)
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if self.verbose >= 2:
            print("\nBluff Timing Statistics:")
            print("\nSuccessful Bluffs:")
            print(success_means)
            print("\nFailed Bluffs:")
            print(failed_means)
        
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
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        fig.suptitle('Adaptive Agent Learning Analysis', fontsize=14, fontweight='bold', y=0.95)
        
        # Plot 1: Learning Progression Over Time
        # Add move number within each game
        df_exploded['move_number'] = df_exploded.groupby('game_id').cumcount() + 1
        
        # Sort by game and move number for continuous progression
        df_exploded = df_exploded.sort_values(['game_id', 'move_number'])
        df_exploded['global_move'] = range(1, len(df_exploded) + 1)
        
        # Bin data into intervals for cleaner visualization
        n_bins = 20  # Adjust number of bins as needed
        df_exploded['bin'] = pd.qcut(df_exploded['global_move'], n_bins, labels=False)
        
        # Calculate binned statistics
        df_binned = df_exploded.groupby('bin').agg({
            'global_move': ['mean', 'std'],
            'predicted_bluff_rate': ['mean', 'std'],
            'actual_bluff_rate': ['mean', 'std'],
            'bluff_threshold': ['mean', 'std']
        })
        
        # Flatten column names
        df_binned.columns = ['_'.join(col).strip() for col in df_binned.columns.values]
        
        # Plot smoothed trends
        ax1.plot(df_binned['global_move_mean'], df_binned['predicted_bluff_rate_mean'],
                label='Predicted Bluff Rate', color='blue', linewidth=2)
        ax1.plot(df_binned['global_move_mean'], df_binned['actual_bluff_rate_mean'],
                label='Actual Bluff Rate', color='orange', linewidth=2)
        ax1.plot(df_binned['global_move_mean'], df_binned['bluff_threshold_mean'],
                label='Bluff Threshold', color='green', linewidth=2)
        
        # Add light scatter points for raw values
        ax1.scatter(df_exploded['global_move'], df_exploded['predicted_bluff_rate'],
                   alpha=0.1, color='blue', s=5)
        ax1.scatter(df_exploded['global_move'], df_exploded['actual_bluff_rate'],
                   alpha=0.1, color='orange', s=5)
        ax1.scatter(df_exploded['global_move'], df_exploded['bluff_threshold'],
                   alpha=0.1, color='green', s=5)
        
        # Add shaded confidence intervals where std > 0
        for col, color in [('predicted_bluff_rate', 'blue'), 
                         ('actual_bluff_rate', 'orange'),
                         ('bluff_threshold', 'green')]:
            std_col = f'{col}_std'
            mean_col = f'{col}_mean'
            mask = df_binned[std_col] > 0
            if mask.any():
                ax1.fill_between(df_binned.loc[mask, 'global_move_mean'],
                               df_binned.loc[mask, mean_col] - df_binned.loc[mask, std_col],
                               df_binned.loc[mask, mean_col] + df_binned.loc[mask, std_col],
                               alpha=0.1, color=color)
        
        # Add vertical lines to separate games
        for game_end in df_exploded.groupby('game_id')['global_move'].max():
            ax1.axvline(x=game_end, color='gray', linestyle='--', alpha=0.2)
        
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
        """Analyze and plot LLM agent's reasoning patterns."""
        df = self.to_dataframe()
        
        if df.empty:
            if self.verbose >= 1:
                print("Warning: No data available for LLM reasoning plot")
            return
        
        # Filter for LLM agents (note: agent type is stored without 'Agent' suffix)
        df_exploded = df[df['agent_type'] == 'LLM'].explode(['decision_type', 'successful_move'])
        
        if df_exploded.empty:
            if self.verbose >= 1:
                print("Warning: No valid LLM reasoning data")
            return
        
        # Convert boolean successful_move to numeric
        df_exploded['successful_move'] = df_exploded['successful_move'].map({True: 1, False: 0})
        
        # Create figure with three subplots
        fig = plt.figure(figsize=(15, 15))
        gs = plt.GridSpec(3, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, :])  # Decision type distribution
        ax2 = fig.add_subplot(gs[1, :])  # Success rate by decision type
        ax3 = fig.add_subplot(gs[2, 0])  # Success rate over time
        ax4 = fig.add_subplot(gs[2, 1])  # Decision type evolution
        
        fig.suptitle('LLM Agent Decision Analysis', fontsize=14, fontweight='bold', y=0.95)
        
        # 1. Decision Type Distribution (Pie Chart)
        decision_counts = df_exploded['decision_type'].value_counts()
        total_decisions = len(df_exploded)
        sizes = [count/total_decisions * 100 for count in decision_counts]
        ax1.pie(sizes, labels=decision_counts.index,
                autopct='%1.1f%%', startangle=90)
        ax1.set_title('Distribution of Decision Types')
        
        # 2. Success Rate by Decision Type (Bar Chart)
        success_rates = df_exploded.groupby('decision_type')['successful_move'].agg(['mean', 'count', 'std'])
        success_rates = success_rates.sort_values('mean', ascending=False)
        
        bars = ax2.bar(range(len(success_rates)), success_rates['mean'])
        
        # Add error bars
        yerr = success_rates['std'] / np.sqrt(success_rates['count'])  # Standard error
        ax2.errorbar(range(len(success_rates)), success_rates['mean'], 
                    yerr=yerr, fmt='none', color='black', capsize=5)
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            count = success_rates['count'].iloc[i]
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}\n(n={count})',
                    ha='center', va='bottom')
        
        ax2.set_title('Success Rate by Decision Type')
        ax2.set_ylabel('Success Rate')
        ax2.set_ylim(0, 1)
        ax2.set_xticks(range(len(success_rates)))
        ax2.set_xticklabels(success_rates.index, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # 3. Success Rate Over Time
        df_exploded['move_number'] = df_exploded.groupby('game_id').cumcount() + 1
        window_size = 10
        rolling_success = df_exploded.groupby('move_number')['successful_move'].mean().rolling(window=window_size).mean()
        
        ax3.plot(rolling_success.index, rolling_success.values, 'b-', label=f'{window_size}-Move Average')
        ax3.scatter(df_exploded['move_number'], df_exploded['successful_move'], 
                   alpha=0.2, color='gray', label='Individual Moves')
        ax3.set_title('Success Rate Over Time')
        ax3.set_xlabel('Move Number')
        ax3.set_ylabel('Success Rate')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Decision Type Evolution
        pivot_data = pd.crosstab(df_exploded['move_number'], df_exploded['decision_type'], normalize='index')
        pivot_data.plot(kind='area', stacked=True, ax=ax4, alpha=0.7)
        ax4.set_title('Decision Type Evolution')
        ax4.set_xlabel('Move Number')
        ax4.set_ylabel('Proportion')
        ax4.grid(True, alpha=0.3)
        ax4.legend(title='Decision Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Print statistics if verbose
        if self.verbose >= 2:
            print("\nLLM Reasoning Statistics:")
            print("\nDecision Type Distribution:")
            print(decision_counts)
            print("\nSuccess Rates by Decision Type:")
            print(success_rates)
            print("\nOverall Success Rate:", df_exploded['successful_move'].mean())
        
        plt.tight_layout()
        suffix = f"_{plt_suffix}" if plt_suffix else ""
        plt.savefig(f'{output_dir}/llm_reasoning{suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_reports(self, output_dir: str = 'reports', plt_suffix: str = ''):
        """Generate comprehensive performance reports"""
        import os
        
        # Create base output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # If plt_suffix is provided, create a subdirectory for this run
        if plt_suffix:
            output_dir = os.path.join(output_dir, plt_suffix.strip('_'))
            os.makedirs(output_dir, exist_ok=True)
        
        df = self.to_dataframe()
        agent_types = sorted(df['agent_type'].unique())
        
        # Generate all plots
        self.plot_win_rates(output_dir, '')
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
