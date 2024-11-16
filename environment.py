import random
from typing import List, Dict, Optional
import time
import pandas as pd

class Environment:
    def __init__(self, agents: List, verbose: int = 0, lives: int = 2, num_dice: int = 5, num_faces: int = 6):
        self.original_agents = agents.copy()  # Keep original list of agents
        self.agents = agents.copy()  # Working copy that might get modified during game
        self.num_players = len(agents)
        self.players = [{'lives': lives, 'dice': [random.randint(1, num_faces) for _ in range(num_dice)]} for _ in range(self.num_players)]
        self.last_move = None
        self.current_player_idx = random.randint(0, self.num_players - 1)
        self.verbose = verbose
        
        # Set initial dice for each agent
        for i, agent in enumerate(self.agents):
            agent.set_dice(self.players[i]['dice'])

    def get_last_move(self) -> Optional[Dict]:
        return self.last_move

    def get_num_players(self) -> int:
        return self.num_players

    def get_num_dice_in_play(self) -> int:
        return sum(len(player['dice']) for player in self.players)

    def get_valid_moves(self) -> List[Dict]:
        valid_moves = []
        total_dice = self.get_num_dice_in_play()
        
        if self.last_move:
            last_quantity = self.last_move['quantity']
            last_face_value = self.last_move['face_value']
            
            # Same quantity, higher face value
            for face in range(last_face_value + 1, 7):
                valid_moves.append({'quantity': last_quantity, 'face_value': face})
            
            # Higher quantity, any face value
            for quantity in range(last_quantity + 1, total_dice + 1):
                for face in range(1, 7):
                    valid_moves.append({'quantity': quantity, 'face_value': face})
            
            # Can always call bluff if there's a previous move
            valid_moves.append({'quantity': 0, 'face_value': 0, 'bluff': True})
        else:
            # First move of the game - can bid any quantity and face value
            for quantity in range(1, total_dice + 1):
                for face in range(1, 7):
                    valid_moves.append({'quantity': quantity, 'face_value': face})
        
        return valid_moves

    def get_game_state(self) -> Dict:
        return {
            'last_bid': self.get_last_move(),
            'valid_moves': self.get_valid_moves(),
        }

    def make_move(self, player_idx: int, move: Dict):
        self.last_move = move
        self.current_player_idx = (self.current_player_idx + 1) % self.num_players

    def call_bluff(self, player_idx: int):
        # Count the actual number of dice with the last bid's face value
        last_face_value = self.last_move['face_value']
        last_quantity = self.last_move['quantity']
        
        # Create a dataframe showing counts of each face value
        all_dice = [dice for player in self.players for dice in player['dice']]
        dice_counts = {i: all_dice.count(i) for i in range(1, 7)}
        actual_count = dice_counts[last_face_value]
        
        if self.verbose >= 2:
            df = pd.DataFrame([dice_counts], columns=range(1, 7), index=['counts'])
            print("\nCurrent dice counts:")
            print(df)
            total_dice_count = df.sum().sum()
            print(f"Total dice count: {total_dice_count}")
            print(f"\nActual count of {last_face_value}s: {actual_count}")
            print(f"Last bid was: {last_quantity} {last_face_value}")
        
        # If the bid was a lie (actual count is less than bid), the last bidder loses a life
        # If the bid was true, the challenger loses a life
        loser_idx = (self.current_player_idx - 1) % self.num_players if actual_count < last_quantity else player_idx
        loser_agent = self.agents[loser_idx]
        
        if self.verbose >= 2:
            print(f"{loser_agent.name} loses a life!")
        
        self.eliminate_player(loser_idx)
        self.last_move = None  # Reset last move for new round
        
        # Reroll dice for all players
        for i, player in enumerate(self.players):
            player['dice'] = [random.randint(1, 6) for _ in range(5)]
            if i < len(self.agents):  # Make sure agent still exists
                self.agents[i].set_dice(player['dice'])

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

    def play_game(self):
        if self.verbose >= 2:
            print("\nStarting new game of Liar's Dice!")
            print("Each agent starts with 5 dice and 2 lives.")
        
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
                self.call_bluff(self.current_player_idx)
            else:
                if self.verbose >= 2:
                    print(f"{current_agent.color}{current_agent.name} bids {move['quantity']} {move['face_value']}\033[0m")
                self.make_move(self.current_player_idx, move)
            
            # Add delay between moves
            if self.verbose >= 2:
                time.sleep(2)
        
        winner = self.agents[0]
        if self.verbose >= 1:
            print(f"\nGame Over! {winner.color}{winner.name} wins the game!\033[0m")
        return winner

    def is_game_over(self) -> bool:
        return self.num_players == 1

    def reset(self):
        """Reset the environment to its initial state with agents in original positions."""
        self.agents = self.original_agents.copy()  # Restore original agent order
        self.num_players = len(self.agents)
        self.players = [{'lives': 2, 'dice': [random.randint(1, 6) for _ in range(5)]} for _ in range(self.num_players)]
        self.last_move = None
        self.current_player_idx = random.randint(0, self.num_players - 1)
        
        # Reset dice for each agent
        for i, agent in enumerate(self.agents):
            agent.set_dice(self.players[i]['dice'])
        
        return self
