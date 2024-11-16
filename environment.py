import random
from typing import List, Dict, Optional
import time
import pandas as pd
from collections import defaultdict
from metrics import GameMetrics

class Environment:
    def __init__(self, agents: List, verbose: int = 0, lives: int = 2, num_dice: int = 5, sleep_time: float = 5.0):
        self.original_agents = agents.copy()  # Keep original list of agents
        self.agents = agents.copy()  # Working copy that might get modified during game
        self.num_players = len(agents)
        self.players = [{'lives': lives, 'dice': [random.randint(1, 6) for _ in range(num_dice)]} for _ in range(self.num_players)]
        self.last_move = None
        self.current_player_idx = random.randint(0, self.num_players - 1)
        self.verbose = verbose
        self.sleep_time = sleep_time
        
        # Set initial dice for each agent
        for i, agent in enumerate(self.agents):
            agent.set_dice(self.players[i]['dice'])
        
        # Metrics tracking
        self._bid_quantities = defaultdict(list)  # Track bid quantities per agent
        self._bluff_calls = defaultdict(int)     # Track bluff calls per agent
        self._successful_bluffs = defaultdict(int)
        self._failed_bluffs = defaultdict(int)
        self._successful_catches = defaultdict(int)
        self._failed_catches = defaultdict(int)
        self._elimination_order = []  # Track when agents are eliminated
        self._rounds = 0
    
    def get_last_move(self) -> Optional[Dict]:
        return self.last_move

    def get_num_players(self) -> int:
        return self.num_players

    def get_num_dice_in_play(self) -> int:
        return sum(len(player['dice']) for player in self.players)

    def get_valid_moves(self) -> List[Dict]:
        """Get list of valid moves given the current game state."""
        valid_moves = []
        total_dice = self.get_num_dice_in_play()
        
        if self.last_move:
            last_quantity = self.last_move['quantity']
            last_face_value = self.last_move['face_value']
            
            # Higher quantity
            for quantity in range(last_quantity + 1, total_dice + 1):
                valid_moves.append({
                    'quantity': quantity,
                    'face_value': last_face_value,
                    'bluff': False
                })
            
            # Same quantity, higher face value
            for face in range(last_face_value + 1, 7):
                valid_moves.append({
                    'quantity': last_quantity,
                    'face_value': face,
                    'bluff': False
                })
            
            # Higher quantity, any face value
            for quantity in range(last_quantity + 1, total_dice + 1):
                for face in range(1, 7):
                    valid_moves.append({
                        'quantity': quantity,
                        'face_value': face,
                        'bluff': False
                    })
            
            # Can always call bluff if there's a previous move
            valid_moves.append({
                'quantity': 0,
                'face_value': 0,
                'bluff': True
            })
        else:
            # First move of the game - can bid any quantity and face value
            for quantity in range(1, total_dice + 1):
                for face in range(1, 7):
                    valid_moves.append({
                        'quantity': quantity,
                        'face_value': face,
                        'bluff': False
                    })
        
        return valid_moves

    def get_game_state(self) -> Dict:
        """Get the current game state."""
        # Get total number of dice in play
        total_dice = sum(len(player['dice']) for player in self.players)
        
        # Get valid moves
        valid_moves = []
        if self.last_move is None:
            # First move - can bid any quantity and face value
            for face in range(1, 7):
                for quantity in range(1, total_dice + 1):
                    valid_moves.append({
                        'quantity': quantity,
                        'face_value': face,
                        'bluff': False
                    })
        else:
            # Can either increase quantity of same face, or increase face value
            last_quantity = self.last_move['quantity']
            last_face = self.last_move['face_value']
            
            # Add bluff call as a valid move
            valid_moves.append({
                'quantity': 0,
                'face_value': 0,
                'bluff': True
            })
            
            # Add higher quantity bids
            for quantity in range(last_quantity + 1, total_dice + 1):
                valid_moves.append({
                    'quantity': quantity,
                    'face_value': last_face,
                    'bluff': False
                })
            
            # Add higher face value bids
            for face in range(last_face + 1, 7):
                for quantity in range(last_quantity, total_dice + 1):
                    valid_moves.append({
                        'quantity': quantity,
                        'face_value': face,
                        'bluff': False
                    })
        
        return {
            'last_move': self.last_move,
            'valid_moves': valid_moves,
            'total_dice': total_dice,
            'players': [{'lives': p['lives'], 'num_dice': len(p['dice'])} for p in self.players],
            'current_player': self.current_player_idx
        }

    def make_move(self, player_idx: int, move: Dict):
        """Make a move in the game."""
        self.last_move = move
        self.current_player_idx = (self.current_player_idx + 1) % self.num_players
        
        # Track metrics
        agent_name = self.agents[player_idx].name
        self._bid_quantities[agent_name].append(move['quantity'])
        self._rounds += 1
        
        if self.verbose >= 2:
            print(f"\nTracking move for {agent_name}:")
            print(f"Bid quantities: {self._bid_quantities[agent_name]}")

    def call_bluff(self, caller_idx: int):
        """Handle a bluff call."""
        if self.last_move is None:
            return
        
        # Count the actual number of dice showing the bid face value
        bid_face = self.last_move['face_value']
        bid_quantity = self.last_move['quantity']
        actual_count = sum(dice.count(bid_face) for dice in [p['dice'] for p in self.players])
        
        # Get the indices of the caller and the last bidder
        last_bidder_idx = (caller_idx - 1) % self.num_players
        caller_name = self.agents[caller_idx].name
        bidder_name = self.agents[last_bidder_idx].name
        
        if self.verbose >= 2:
            print(f"\nBluff call: {caller_name} calling {bidder_name}'s bluff")
            print(f"Actual count of {bid_face}s: {actual_count}")
            print(f"Last bid was: {bid_quantity} {bid_face}s")
        
        # Track bluff call
        self._bluff_calls[caller_name] += 1
        
        # Determine who loses a life
        if actual_count >= bid_quantity:
            # Caller was wrong (there were enough dice)
            self.players[caller_idx]['lives'] -= 1
            self._failed_catches[caller_name] += 1
            self._successful_bluffs[bidder_name] += 1
            loser_idx = caller_idx
            if self.verbose >= 2:
                print(f"{caller_name} was wrong! Failed catch recorded.")
                print(f"New failed catches: {self._failed_catches[caller_name]}")
                print(f"New successful bluffs for {bidder_name}: {self._successful_bluffs[bidder_name]}")
        else:
            # Caller was right (not enough dice)
            self.players[last_bidder_idx]['lives'] -= 1
            self._successful_catches[caller_name] += 1
            self._failed_bluffs[bidder_name] += 1
            loser_idx = last_bidder_idx
            if self.verbose >= 2:
                print(f"{caller_name} was right! Successful catch recorded.")
                print(f"New successful catches: {self._successful_catches[caller_name]}")
                print(f"New failed bluffs for {bidder_name}: {self._failed_bluffs[bidder_name]}")
        
        # Check if player is eliminated
        if self.players[loser_idx]['lives'] <= 0:
            self._elimination_order.append(self.agents[loser_idx].name)
            if self.verbose >= 2:
                print(f"{self.agents[loser_idx].name} eliminated! Added to elimination order.")
                print(f"Current elimination order: {self._elimination_order}")
            
            # Remove the player
            self.players.pop(loser_idx)
            self.agents.pop(loser_idx)
            self.num_players -= 1
            
            # Adjust current player index if needed
            if loser_idx <= self.current_player_idx:
                self.current_player_idx = max(0, self.current_player_idx - 1)
        
        self.last_move = None  # Reset last move for new round
        
        # Reroll dice for all players
        for i in range(len(self.players)):
            self.players[i]['dice'] = [random.randint(1, 6) for _ in range(5)]
            self.agents[i].set_dice(self.players[i]['dice'])
        
        # Update current player to be after the loser
        self.current_player_idx = (loser_idx + 1) % self.num_players

    def eliminate_player(self, player_idx: int):
        self.players[player_idx]['lives'] -= 1
        agent = self.agents[player_idx]
        if self.players[player_idx]['lives'] <= 0:
            if self.verbose >= 2:
                print(f"\033[91m{agent.name} has been eliminated!\033[0m")
            self.players.pop(player_idx)
            self.agents.pop(player_idx)  # Only remove from working copy, not original
            self.num_players -= 1
            if player_idx <= self.current_player_idx:
                self.current_player_idx = max(0, self.current_player_idx - 1)

    def get_metrics(self) -> GameMetrics:
        """Get metrics for the current game."""
        winner = next(agent.name for i, agent in enumerate(self.agents) 
                     if self.players[i]['lives'] > 0)
        
        # Calculate average bids
        avg_quantities = {
            agent: sum(bids)/len(bids) if bids else 0 
            for agent, bids in self._bid_quantities.items()
        }
        
        # Calculate bluff rates
        total_moves = {
            agent: len(self._bid_quantities[agent]) + self._bluff_calls[agent]
            for agent in self._bid_quantities.keys()
        }
        bluff_rates = {
            agent: self._bluff_calls[agent] / total_moves[agent] if total_moves[agent] > 0 else 0
            for agent in total_moves.keys()
        }
        
        # Calculate survival time (when they were eliminated)
        survival_times = {}
        for idx, agent in enumerate(self._elimination_order):
            survival_times[agent] = self._rounds - (len(self._elimination_order) - idx - 1)
        # Winner survived the whole game
        survival_times[winner] = self._rounds
        
        return GameMetrics(
            winner=winner,
            rounds=self._rounds,
            final_lives={agent.name: self.players[i]['lives'] for i, agent in enumerate(self.agents)},
            bluff_calls=dict(self._bluff_calls),
            successful_bluffs=dict(self._successful_bluffs),
            failed_bluffs=dict(self._failed_bluffs),
            successful_catches=dict(self._successful_catches),
            failed_catches=dict(self._failed_catches),
            avg_bid_quantity=avg_quantities,
            bluff_rate=bluff_rates,
            survival_time=survival_times
        )

    def play_game(self):
        if self.verbose >= 2:
            print("\n" + "="*50)
            print(f"{' STARTING NEW GAME ':=^50}")
            print("="*50)
            print("\nPlayers:")
            for i, agent in enumerate(self.agents):
                print(f"{agent.color}{agent.name}: {len(self.players[i]['dice'])} dice, {self.players[i]['lives']} lives\033[0m")
            print("\nLet the game begin!")
            print("-"*50)
        
        while not self.is_game_over():
            current_agent = self.agents[self.current_player_idx]
            if self.verbose >= 2:
                print(f"\n{current_agent.color}{current_agent.name}'s turn. Lives: {self.players[self.current_player_idx]['lives']}.")
                print(f"Their dice: {sorted(self.players[self.current_player_idx]['dice'])}\033[0m")
            
            # Get move from current agent
            move = current_agent.make_move(self)
            
            # Apply the move
            if move.get('bluff', False) or (move['quantity'] == 0 and move['face_value'] == 0):
                if self.verbose >= 2:
                    print(f"\033[91m{current_agent.name} calls bluff!\033[0m")
                    if 'reasoning' in move:
                        print(f"{current_agent.color}Reasoning: {move['reasoning']}\033[0m")
                self.call_bluff(self.current_player_idx)
            else:
                if self.verbose >= 2:
                    print(f"{current_agent.color}{current_agent.name} bids {move['quantity']} {move['face_value']}s\033[0m")
                    if 'reasoning' in move:
                        print(f"{current_agent.color}Reasoning: {move['reasoning']}\033[0m")
                self.make_move(self.current_player_idx, move)
            
            # Add delay between moves
            if self.verbose >= 2:
                time.sleep(self.sleep_time)
        
        # Game is over, get winner
        winner = self.agents[0]
        if self.verbose >= 1:
            print(f"\nGame Over! {winner.color}{winner.name} wins the game!\033[0m")
        return winner

    def is_game_over(self) -> bool:
        """Check if the game is over."""
        active_players = sum(1 for p in self.players if p['lives'] > 0)
        if active_players == 1:
            # Find the winner and add to elimination order if not already there
            winner = next(agent.name for i, agent in enumerate(self.agents) 
                        if self.players[i]['lives'] > 0)
            if winner not in self._elimination_order:
                self._elimination_order.append(winner)
        return active_players <= 1

    def reset(self):
        """Reset the game state."""
        # Restore original agents
        self.agents = self.original_agents.copy()
        self.num_players = len(self.agents)
        
        if self.verbose >= 2:
            print("\nPre-shuffle positions:")
            for i, agent in enumerate(self.agents):
                print(f"{i}: {agent.name}")
        
        # Shuffle agent order
        random.shuffle(self.agents)
        
        if self.verbose >= 2:
            print("\nPost-shuffle positions:")
            for i, agent in enumerate(self.agents):
                print(f"{i}: {agent.name}")
        
        # Reset game state
        self.players = []
        for _ in range(self.num_players):
            self.players.append({
                'dice': [random.randint(1, 6) for _ in range(5)],
                'lives': 2
            })
        self.current_player_idx = 0
        self.last_move = None
        
        # Reset metrics
        self._bid_quantities.clear()
        self._bluff_calls.clear()
        self._successful_bluffs.clear()
        self._failed_bluffs.clear()
        self._successful_catches.clear()
        self._failed_catches.clear()
        self._elimination_order.clear()
        self._rounds = 0
        
        # Initialize metrics for each agent
        for agent in self.agents:
            self._bid_quantities[agent.name] = []
            self._bluff_calls[agent.name] = 0
            self._successful_bluffs[agent.name] = 0
            self._failed_bluffs[agent.name] = 0
            self._successful_catches[agent.name] = 0
            self._failed_catches[agent.name] = 0
            
            if self.verbose >= 2:
                print(f"\nInitialized metrics for {agent.name}")
                print(f"Bluff calls: {self._bluff_calls[agent.name]}")
                print(f"Successful bluffs: {self._successful_bluffs[agent.name]}")
                print(f"Failed bluffs: {self._failed_bluffs[agent.name]}")
        
        # Set initial dice for each agent
        for i, agent in enumerate(self.agents):
            agent.set_dice(self.players[i]['dice'])
            if self.verbose >= 2:
                print(f"\n{agent.name} at position {i} got dice: {self.players[i]['dice']}")
        
        return self
