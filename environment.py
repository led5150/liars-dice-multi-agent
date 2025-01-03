import random
from typing import List, Dict, Optional
import time
from time import sleep
import pandas as pd
from collections import defaultdict
import os
from metrics import GameMetrics
from agent import InformedAgent, AdaptiveAgent, LLMAgent

class Environment:
    def __init__(self, agents: List, verbose: int = 0, lives: int = 2, num_dice: int = 5, sleep_time: float = 5.0, clear_screen: bool = False):
        self.original_agents = agents.copy()  # Keep original list of agents
        self.agents = agents.copy()  # Working copy that might get modified during game
        self.num_players = len(agents)
        self.players = [{'lives': lives, 'dice': [random.randint(1, 6) for _ in range(num_dice)]} for _ in range(self.num_players)]
        self.last_move = None
        self.current_player_idx = random.randint(0, self.num_players - 1)
        self.verbose = verbose
        self.sleep_time = sleep_time
        self.clear_screen = clear_screen
        self.game_number = 0  # Track current game number
        
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
        
        # New metrics for enhanced analysis
        self._prev_bid_quantity_successful_bluff = defaultdict(list)
        self._prev_bid_quantity_failed_bluff = defaultdict(list)
        self._predicted_probability = defaultdict(list)
        self._actual_outcome = defaultdict(list)
        self._bluff_threshold = defaultdict(list)
        self._predicted_bluff_rate = defaultdict(list)
        self._actual_bluff_rate = defaultdict(list)
        self._decision_type = defaultdict(list)
        self._successful_move = defaultdict(list)
    
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
        
        # if self.verbose >= 2:
        #     print("\nGenerating valid moves:")
        #     print(f"Total dice in play: {total_dice}")
        #     if self.last_move:
        #         print(f"Last move: {self.last_move}")
        
        if self.last_move:
            last_quantity = self.last_move['quantity']
            last_face = self.last_move['face_value']
            
            # Can always call bluff if there's a previous move
            valid_moves.append({
                'quantity': 0,
                'face_value': 0,
                'bluff': True
            })
            
            # For the same face value, you can only bid higher quantities
            for quantity in range(last_quantity + 1, total_dice + 1):
                valid_moves.append({
                    'quantity': quantity,
                    'face_value': last_face,
                    'bluff': False
                })
                
            # For different face values, you must bid higher quantities
            for quantity in range(last_quantity + 1, total_dice + 1):
                for face in range(1, 7):
                    if face != last_face:
                        valid_moves.append({
                            'quantity': quantity,
                            'face_value': face,
                            'bluff': False
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
        valid_moves = self.get_valid_moves()
        
        return {
            'last_move': self.last_move,
            'valid_moves': valid_moves,
            'total_dice': total_dice,
            'players': [{'lives': p['lives'], 'num_dice': len(p['dice'])} for p in self.players],
            'current_player': self.current_player_idx
        }

    # def make_move(self, player_idx: int, move: Dict):
    #     """Make a move in the game."""
    #     self.last_move = move
    #     self.current_player_idx = (self.current_player_idx + 1) % self.num_players
        
    #     # Track metrics
    #     agent_name = self.agents[player_idx].name
    #     self._rounds += 1
        


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
        else:
            # Caller was right (not enough dice)
            self.players[last_bidder_idx]['lives'] -= 1
            self._successful_catches[caller_name] += 1
            self._failed_bluffs[bidder_name] += 1
            loser_idx = last_bidder_idx
            if self.verbose >= 2:
                print(f"{caller_name} was right! Successful catch recorded.")

        
        # Check if player is eliminated
        if self.players[loser_idx]['lives'] <= 0:
            self._elimination_order.append(self.agents[loser_idx].name)

            
            # Remove the player
            self.players.pop(loser_idx)
            self.agents.pop(loser_idx)
            self.num_players -= 1
            
            # Adjust current player index if needed
            if loser_idx <= self.current_player_idx:
                self.current_player_idx = max(0, self.current_player_idx - 1)
                
        # Reroll dice for all players
        for i in range(len(self.players)):
            self.players[i]['dice'] = [random.randint(1, 6) for _ in range(5)]
            self.agents[i].set_dice(self.players[i]['dice'])
        
        # Update current player to be after the loser
        self.current_player_idx = (loser_idx + 1) % self.num_players

    def eliminate_player(self, player_idx: int, winner_idx: int):
        """Eliminate a player from the game."""
        agent = self.agents[player_idx]
        if self.verbose >= 2:
            print(f"\033[91m{agent.name} has been eliminated!\033[0m")
        
        # Add to elimination order if not already there
        if agent.name not in self._elimination_order:
            self._elimination_order.append(agent.name)
        
        # # Store winner before removing player
        # winner_agent = self.agents[winner_idx]
        
        # Remove player from game
        self.players.pop(player_idx)
        self.agents.pop(player_idx)  # Remove from working copy
        self.num_players -= 1
        
        # Winner gets to play next, we need to ensure their index is valid
        self.current_player_idx = max(0, self.current_player_idx - 1)
        

    def _handle_bluff_call(self) -> bool:
        """
        Handle a bluff call and determine if it was successful.
        Returns True if the bluff call was successful (previous player was bluffing).
        """
        if not self.last_move:
            return False  # No previous move to check
            
        # Get total count of dice matching the last bid
        total_count = 0
        for player in self.players:
            total_count += sum(1 for die in player['dice'] if die == self.last_move['face_value'])
            
        if self.verbose >= 1:
            print(f"\nTotal {self.last_move['face_value']}s in play: {total_count}")
            print(f"Last bid: {self.last_move['quantity']} {self.last_move['face_value']}s")
            print("-" * 50)
            
        # If the actual count is less than the bid, the bluff call was successful
        bluff_call_successful = total_count < self.last_move['quantity']
        
        # Mark the move as a bluff or not for learning
        self.last_move['was_bluff'] = bluff_call_successful
        
        # Notify agents of the updated game state
        for agent in self.agents:
            if hasattr(agent, 'update_from_game_state'):
                agent.update_from_game_state(self.get_game_state())
        
        # Calculate bluff rate for metrics
        prev_player = self.agents[self._get_previous_player()].name
        total_moves = (self._successful_bluffs[prev_player] + 
                      self._failed_bluffs[prev_player])
        if total_moves > 0:
            bluff_rate = self._successful_bluffs[prev_player] / total_moves
            self._actual_bluff_rate[prev_player].append(bluff_rate)
        
        # Determine which player loses a life
        if bluff_call_successful:
            # Previous player was bluffing, they lose a life
            loser_idx = self._get_previous_player()
            winner_idx = self.current_player_idx
            if self.verbose >= 1:
                print(f"{self.agents[loser_idx].color}{self.agents[loser_idx].name} tried to bluff!\033[0m")
                print(f"{self.agents[self.current_player_idx].color}{self.agents[self.current_player_idx].name} Called their bluff successfully!\033[0m")
        else:
            # Bluff call was wrong, current player loses a life
            loser_idx = self.current_player_idx
            winner_idx = self._get_previous_player()
            if self.verbose >= 1:
                print(f"{self.agents[winner_idx].color}{self.agents[winner_idx].name} tried to bluff!\033[0m")
                print(f"{self.agents[self.current_player_idx].color}{self.agents[self.current_player_idx].name}'s bluff call was wrong!\033[0m")
                
        # Reduce life count and eliminate if necessary
        self.players[loser_idx]['lives'] -= 1
        if self.verbose >= 1:
            print(f"\033[91m{self.agents[loser_idx].name} lost a life! Lives remaining: {self.players[loser_idx]['lives']}\033[0m")
            print("-" * 50)
            
        if self.players[loser_idx]['lives'] <= 0:
            self.eliminate_player(loser_idx, winner_idx)
            

        # Move to next player (skipping eliminated players)
        if len(self.agents) > 1:  # Only update if game isn't over
            # Reroll dice for next round
            for player in self.players:
                player['dice'] = [random.randint(1, 6) for _ in range(5)]  # Always 5 dice
                
            # Update agent dice information
            for i, agent in enumerate(self.agents):
                agent.set_dice(self.players[i]['dice'])
            
        return bluff_call_successful

    def _update_metrics(self, agent_name: str, move: Dict, was_successful: bool):
        """Update metrics after a move."""
        # Track bid quantities for all valid moves with a quantity
        if 'quantity' in move:
            self._bid_quantities[agent_name].append(move['quantity'])
        
        # Track bluff calls and outcomes
        if move.get('bluff', False):
            self._bluff_calls[agent_name] += 1
            prev_bid = self.last_move['quantity'] if self.last_move else None

            if was_successful:
                # Current player successfully caught a bluff
                self._successful_catches[agent_name] += 1
                if prev_bid is not None:
                    self._prev_bid_quantity_successful_bluff[agent_name].append(prev_bid)
                # Previous player's bluff was caught
                prev_player = self.agents[self._get_previous_player()].name
                self._successful_bluffs[prev_player] += 1
            else:
                # Current player failed to catch a bluff
                self._failed_catches[agent_name] += 1
                if prev_bid is not None:
                    self._prev_bid_quantity_failed_bluff[agent_name].append(prev_bid)
                # Previous player successfully bluffed
                prev_player = self.agents[self._get_previous_player()].name
                self._failed_bluffs[prev_player] += 1
        
        # Track prediction metrics
        if 'predicted_probability' in move:
            self._predicted_probability[agent_name].append(move['predicted_probability'])
            self._actual_outcome[agent_name].append(float(was_successful))
            
        # Track adaptive learning metrics
        if 'predicted_bluff_rate' in move:
            self._predicted_bluff_rate[agent_name].append(move['predicted_bluff_rate'])
            if 'actual_bluff_rate' in move:
                self._actual_bluff_rate[agent_name].append(move['actual_bluff_rate'])
        
        # Track bluff threshold
        if 'bluff_threshold' in move:
            self._bluff_threshold[agent_name].append(move['bluff_threshold'])
            
        # Track decision type and success
        if 'reasoning' in move:
            self._decision_type[agent_name].append(move['reasoning'])
            self._successful_move[agent_name].append(was_successful)

    def get_metrics(self) -> GameMetrics:
        """Get metrics for the current game."""
        winner = next(agent.name for i, agent in enumerate(self.agents) 
                     if self.players[i]['lives'] > 0)
        
        if self.verbose >= 2:
            print("\nGame Metrics:")
            print(f"Winner: {winner}")
            print(f"Elimination Order: {self._elimination_order}")
            print(f"Final Lives: {[(agent.name, self.players[i]['lives']) for i, agent in enumerate(self.agents)]}")
        
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
            # Each agent survives at least 1 round, and then add additional rounds based on elimination order
            survival_times[agent] = max(1, self._rounds - (len(self._elimination_order) - idx - 1))
        # Winner survived all rounds
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
            survival_time=survival_times,
            elimination_order=self._elimination_order,
            
            # New metrics
            prev_bid_quantity_successful_bluff=dict(self._prev_bid_quantity_successful_bluff),
            prev_bid_quantity_failed_bluff=dict(self._prev_bid_quantity_failed_bluff),
            predicted_probability=dict(self._predicted_probability),
            actual_outcome=dict(self._actual_outcome),
            bluff_threshold=dict(self._bluff_threshold),
            predicted_bluff_rate=dict(self._predicted_bluff_rate),
            actual_bluff_rate=dict(self._actual_bluff_rate),
            decision_type=dict(self._decision_type),
            successful_move=dict(self._successful_move)
        )

    def _get_next_player(self) -> int:
        """Get the index of the next active player."""
        next_idx = (self.current_player_idx + 1) % len(self.agents)
        return next_idx

    def _get_previous_player(self) -> int:
        """Get the index of the previous active player."""
        prev_idx = (self.current_player_idx - 1) % len(self.agents)
        return prev_idx

    def _clear_screen_if_enabled(self):
        """Clear the screen if clear_screen is enabled."""
        if self.clear_screen:
            os.system('clear' if os.name == 'posix' else 'cls')

    def _print_game_state(self):
        """Print the current game state."""
        self._clear_screen_if_enabled()
        print("\nCurrent Game State:")
        print("-" * 50)
        current_agent = self.agents[self.current_player_idx]
        print(f"Current Player: {current_agent.color}{current_agent.name}\033[0m")
        print("\nPlayers:")
        for i, (agent, player) in enumerate(zip(self.agents, self.players)):
            dice_str = ' '.join(str(d) for d in sorted(player['dice']))
            # Add arrow to indicate current player
            current = "→ " if i == self.current_player_idx else "  "
            print(f"{current}{agent.color}{agent.name}\033[0m: Lives={player['lives']}, Dice=[{dice_str}]")
        
        if self.last_move:
            last_player = self.agents[self._get_previous_player()]
            print("\nLast Move:", end=' ')
            if self.last_move.get('bluff', False):
                # print(f"{last_player.color}Called Bluff!\033[0m")
                pass
            else:
                print(f"{last_player.color}Bid: {self.last_move['quantity']} {self.last_move['face_value']}s\033[0m")
        

    def _print_move_outcome(self, move: Dict):
        """Print the outcome of a move."""
        current_agent = self.agents[self.current_player_idx]
        agent_name = current_agent.name
        agent_color = current_agent.color
        current_dice = self.players[self.current_player_idx]['dice']
        if move.get('bluff', False):
            print(f"\n{agent_color}{agent_name} called Liar!!")
            print(f"Reasoning: {move['reasoning']}\033[0m")
        else:
            print(f"\n{agent_color}{agent_name} bids {move['quantity']} {move['face_value']}s")
            dice_str = "[" + ', '.join(str(d) for d in sorted(current_dice)) + "]"
            print(f"Dice: {dice_str}")
            print(f"Reasoning: {move['reasoning']}\033[0m")
            print("-" * 50)

    def reset(self):
        """Reset the game state."""
        # Restore original agents
        self.agents = self.original_agents.copy()
        self.num_players = len(self.agents)
        
        # Reset player states
        self.players = [{'lives': 2, 'dice': [random.randint(1, 6) for _ in range(5)]} 
                       for _ in range(self.num_players)]
        
        # Shuffle positions
        random.shuffle(self.agents)
        
        # Set initial dice for each agent
        for i, agent in enumerate(self.agents):
            agent.set_dice(self.players[i]['dice'])
        
        # Reset game state
        self.last_move = None
        self.current_player_idx = random.randint(0, self.num_players - 1)
        self.game_number = 0  # Reset game number
        
        # Reset metrics
        self._bid_quantities = defaultdict(list)
        self._bluff_calls = defaultdict(int)
        self._successful_bluffs = defaultdict(int)
        self._failed_bluffs = defaultdict(int)
        self._successful_catches = defaultdict(int)
        self._failed_catches = defaultdict(int)
        self._elimination_order = []  # Clear elimination order for new game
        self._rounds = 0
        
        # Reset new metrics
        self._prev_bid_quantity_successful_bluff = defaultdict(list)
        self._prev_bid_quantity_failed_bluff = defaultdict(list)
        self._predicted_probability = defaultdict(list)
        self._actual_outcome = defaultdict(list)
        self._bluff_threshold = defaultdict(list)
        self._predicted_bluff_rate = defaultdict(list)
        self._actual_bluff_rate = defaultdict(list)
        self._decision_type = defaultdict(list)
        self._successful_move = defaultdict(list)

    def play_game(self):
        """Play a single game of Liar's Dice."""
        if self.verbose >= 2:
            print("\n" + "="*50)
            print(f"🎲 GAME {self.game_number + 1} 🎲".center(50))
            print("="*50 + "\n")
        
        while not self.is_game_over():
            if self.verbose >= 2:
                self._print_game_state()
            
            # Get current agent's move
            agent = self.agents[self.current_player_idx]
            move = agent.make_move(self)

            # If this is a bluff call, handle it
            if move.get('bluff', False):
                if self.verbose >= 2:
                    self._print_move_outcome(move)
                    
                # Preserve the adjust_threshold flag in last_move for agent learning
                if self.last_move:
                    self.last_move['adjust_threshold'] = move.get('adjust_threshold', False)
                
                success = self._handle_bluff_call()
                self._update_metrics(agent.name, move, success)
                self.last_move = None
            else:
                # Update game state with new move
                if self.last_move:
                    # Previous move wasn't a bluff since no one called it
                    self.last_move['was_bluff'] = False
                
                move['was_bluff'] = None  # Will be determined by next player's action
                self.last_move = move
                self._update_metrics(agent.name, move, True)  # Move was accepted
                # Print move outcome if verbose
                if self.verbose >= 2:
                    self._print_move_outcome(move)

                self.current_player_idx = self._get_next_player()
                

            
            if self.verbose >= 2:
                sleep(self.sleep_time)
        
        # Find the winner (player with lives remaining)
        winner = next(agent.name for i, agent in enumerate(self.agents) 
                     if self.players[i]['lives'] > 0)
        if self.verbose >= 1:
            print(f"\n🏆 {winner.split('_')[0]} agent {winner} wins! 🏆")
            
        if self.verbose >= 2:
            print("\n" + "="*50)
            print(f"🎲 GAME {self.game_number + 1} OVER 🎲".center(50))
            print("="*50 + "\n")
        if self.verbose >= 2:
            sleep(self.sleep_time)
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
