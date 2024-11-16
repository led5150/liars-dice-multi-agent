from abc import ABC, abstractmethod
import random
from typing import List, Dict, Tuple, Optional
import json
from openai import OpenAI
from huggingface_hub import InferenceClient

# ANSI color codes for agents (nice colors that are easy on the eyes)
AGENT_COLORS = [
    '\033[96m',  # Cyan
    '\033[92m',  # Green
    '\033[94m',  # Blue
    '\033[95m',  # Magenta
    '\033[93m',  # Yellow
]

class LiarsDiceAgent(ABC):
    """Abstract base class for Liar's Dice agents"""
    
    def __init__(self, name: str, color_idx: int = 0):
        self.name = name
        self.dice: List[int] = []
        self.color = AGENT_COLORS[color_idx % len(AGENT_COLORS)]
        
    @abstractmethod
    def make_move(self, environment) -> Dict:
        """
        Make a move based on the current game state
        Returns: Dict with face_value, quantity, and bluff
        """
        pass
    
    def set_dice(self, dice: List[int]):
        """Set the agent's current dice"""
        self.dice = sorted(dice)

class RandomAgent(LiarsDiceAgent):
    """Agent that makes random valid moves"""
    
    def make_move(self, environment) -> Dict:
        game_state = environment.get_game_state()
        last_bid = game_state.get('last_bid')
        
        # 20% chance to call bluff if there's a previous bid
        if last_bid and random.random() < 0.2:
            return {
                'face_value': 0,
                'quantity': 0,
                'bluff': True,
                'reasoning': 'Random decision to call bluff'
            }
        

        # Get valid moves from game state
        valid_moves = game_state.get('valid_moves', [])
        
        # Randomly select one of the valid moves
        move = random.choice(valid_moves)
        
        face_value = move['face_value'] 
        quantity = move['quantity']

        
        return {
            'face_value': face_value,
            'quantity': quantity,
            'bluff': False,
            'reasoning': 'Random valid move'
        }

class InformedAgent(LiarsDiceAgent):
    """Agent that uses probability calculations to make decisions"""
    
    def __init__(self, name: str, color_idx: int = 0):
        super().__init__(name, color_idx)
        from liars_dice_calculator import liars_dice_calc
        self.calc_probability = liars_dice_calc
    
    def make_move(self, environment) -> Dict:
        game_state = environment.get_game_state()
        last_move = game_state.get('last_move')
        valid_moves = game_state.get('valid_moves', [])
        
        P_bluff = 0.0  # Probability that the last bid was a bluff
        
        if last_move:
            # Calculate the probability that the last bid is truthful
            face_value = last_move['face_value']
            quantity = last_move['quantity']
            
            # Number of the bid's face_value dice the agent holds
            my_relevant_dice = self.dice.count(face_value)
            
            # Remaining dice in play
            remaining_dice = game_state['total_dice'] - len(self.dice)
            
            # Required dice from others to make the bid true
            required_successes = quantity - my_relevant_dice
            
            if required_successes <= 0:
                # Agent already holds enough dice to confirm the bid
                P_truthful = 1.0
            else:
                # Calculate probability that at least 'required_successes' dice show 'face_value'
                P_truthful = self.calc_probability(
                    D=remaining_dice,
                    p=1/6,
                    c=required_successes,
                    k=0
                )
            
            # Probability that the last bid was a bluff
            P_bluff = 1 - P_truthful
        
        # Evaluate all valid moves and calculate their probabilities
        move_probabilities = []
        for move in valid_moves:
            face_value = move['face_value']
            quantity = move['quantity']
            
            # Number of the bid's face_value dice the agent holds
            my_relevant_dice = self.dice.count(face_value)
            
            # Remaining dice in play
            remaining_dice = environment.get_num_dice_in_play() - len(self.dice)
            
            # Required dice from others to make the bid true
            required_successes = quantity - my_relevant_dice
            
            if required_successes <= 0:
                probability = 1.0  # Guaranteed to be true since agent has enough dice
            else:
                probability = self.calc_probability(
                    D=remaining_dice,
                    p=1/6,
                    c=required_successes,
                    k=0
                )
            
            move_probabilities.append({
                'move': move,
                'probability': probability
            })
        
        # Select the move with the highest probability
        best_move_entry = max(move_probabilities, key=lambda x: x['probability'], default=None)
        
        # Determine the probability of making the best move
        P_move = best_move_entry['probability'] if best_move_entry else 0.0
        
        # Decision Logic: Choose whichever probability is higher
        if last_move:
            if P_bluff > P_move:
                # Decide to call a bluff
                return {
                    'face_value': 0,
                    'quantity': 0,
                    'bluff': True,
                    'reasoning': f'Chose to call bluff based on higher probability ({P_bluff:.2%}) that the last bid was a bluff.'
                }
            elif P_move > P_bluff:
                # Decide to make the best move
                selected_move = best_move_entry['move']
                return {
                    'face_value': selected_move['face_value'],
                    'quantity': selected_move['quantity'],
                    'bluff': False,
                    'reasoning': f'Chose to make the best move with probability ({P_move:.2%}) of being accurate.'
                }
            else:
                # Probabilities are equal; make a 50/50 decision
                if random.random() < 0.5:
                    # Call bluff
                    return {
                        'face_value': 0,
                        'quantity': 0,
                        'bluff': True,
                        'reasoning': 'Probabilities equal. Decided to call bluff with a 50% chance.'
                    }
                else:
                    # Make the best move
                    selected_move = best_move_entry['move']
                    return {
                        'face_value': selected_move['face_value'],
                        'quantity': selected_move['quantity'],
                        'bluff': False,
                        'reasoning': 'Probabilities equal. Decided to make the best move with a 50% chance.'
                    }
        else:
            if best_move_entry:
                return {
                    'face_value': best_move_entry['move']['face_value'],
                    'quantity': best_move_entry['move']['quantity'],
                    'bluff': False,
                    'reasoning': f'No prior bids, making the best available move with probability {P_move:.2%}.'
                }
            else:
                # No valid moves available; default to calling a bluff
                return {
                    'face_value': 0,
                    'quantity': 0,
                    'bluff': True,
                    'reasoning': 'No valid moves available, calling bluff by default.'
                }

class LLMAgent(LiarsDiceAgent):
    """Agent that uses LLM for decision making"""
    
    def __init__(self, name: str, provider: str = 'openai', model: str = None, api_key: str = None, color_idx: int = 0):
        super().__init__(name, color_idx)
        self.provider = provider
        self.system_prompt = self._load_system_prompt()
        
        if provider == 'openai':
            self.client = OpenAI(api_key=api_key)
            self.model = model or "gpt-3.5-turbo"
        elif provider == 'huggingface':
            self.client = InferenceClient(model=model, token=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _load_system_prompt(self) -> str:
        """Load the system prompt from meta-prompts.txt"""
        with open('meta-prompts.txt', 'r') as f:
            return f.read()
    
    def make_move(self, game_state: Dict) -> Dict:
        """Make a move using the LLM"""
        prompt = self._format_game_state(game_state)
        
        if self.provider == 'openai':
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            move = json.loads(response.choices[0].message.content)
            
        elif self.provider == 'huggingface':
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            response = self.client.chat_completion(
                messages,
                model=self.model,
                temperature=0.7
            )
            move = json.loads(response.choices[0].message.content)
            
        return move
    
    def _format_game_state(self, game_state: Dict) -> str:
        """Format the game state for the LLM prompt"""
        return f"""
Current dice: {self.dice}
History: {json.dumps(game_state.get('history', []))}
Total dice in play: {game_state.get('total_dice', 0)}
Last bid: {json.dumps(game_state.get('last_bid', None))}
""" 