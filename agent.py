from abc import ABC, abstractmethod
import random
from typing import List, Dict, Tuple, Optional
import json
from openai import OpenAI
from huggingface_hub import InferenceClient
from pydantic import BaseModel
from prompts.meta_prompts import (
    NAIVE_SYSTEM_PROMPT, 
    INFORMED_SYSTEM_PROMPT,
    GAME_STATE_PROMPT,
    ERROR_CORRECTION_PROMPT,
    INVALID_RESPONSE_PROMPT
)

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
    
    def __init__(self, name: str, color_idx: int = 0, verbose: int = 0):
        self.name = name
        self.dice: List[int] = []
        self.color = AGENT_COLORS[color_idx % len(AGENT_COLORS)]
        self.verbose = verbose
        self.agent_type = self.__class__.__name__.replace('Agent', '')  # Extract agent type from class name
        
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

    def update_from_game_state(self, game_state: Dict):
        """Update agent's strategy based on the game state"""
        pass

class LiarsMove(BaseModel):
    """Pydantic model for structured output from OpenAI"""
    quantity: int
    face_value: int
    bluff: bool
    reasoning: str

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
        move = random.choice(valid_moves)
        move['reasoning'] = 'Random valid move'
        return move

class InformedAgent(LiarsDiceAgent):
    """Agent that uses probability calculations to make decisions"""
    
    def __init__(self, name: str, color_idx: int = 0):
        super().__init__(name, color_idx)
        from liars_dice_calculator import liars_dice_calc
        self.calc_probability = liars_dice_calc
        self.last_probability = None  # Track last calculated probability
    
    def make_move(self, environment) -> Dict:
        game_state = environment.get_game_state()
        last_move = game_state.get('last_move')
        valid_moves = game_state.get('valid_moves', [])
        

        
        # First, calculate P_bluff for the last move
        P_bluff = 0.0
        if last_move:
            face_value = last_move['face_value']
            quantity = last_move['quantity']
            my_relevant_dice = self.dice.count(face_value)
            required_successes = quantity - my_relevant_dice
            remaining_dice = game_state['total_dice'] - len(self.dice)
            
            if required_successes <= 0:
                # We can verify the bid is true
                P_truthful = 1.0
            else:
                # Calculate probability that at least required_successes dice show face_value
                P_truthful = self.calc_probability(
                    D=remaining_dice,
                    p=1/6,
                    c=required_successes,
                    k=0
                )
            P_bluff = 1 - P_truthful
        
        # Store the probability for metrics
        self.last_probability = P_bluff
        
        # Evaluate all moves and store their probabilities
        move_evaluations = []
        
        for move in valid_moves:
            if move['bluff']:
                move_evaluations.append((move, P_bluff))
                continue
                
            face_value = move['face_value']
            quantity = move['quantity']
            my_relevant_dice = self.dice.count(face_value)
            required_successes = quantity - my_relevant_dice
            remaining_dice = game_state['total_dice'] - len(self.dice)
            
            if required_successes <= 0:
                # We have all the dice we need - this should be our preferred move
                probability = 2.0  # Much higher bonus for having all needed dice
                if my_relevant_dice > quantity:
                    # We have even more than needed!
                    probability = 3.0
            else:
                probability = self.calc_probability(
                    D=remaining_dice,
                    p=1/6,
                    c=required_successes,
                    k=0
                )
            
            # Prefer moves that don't increase quantity too much
            if last_move and quantity > last_move['quantity']:
                probability *= 0.95  # Small penalty for increasing quantity
            
            move_evaluations.append((move, probability))
        
        # Sort moves by probability and show top 5
        move_evaluations.sort(key=lambda x: x[1], reverse=True)

        if self.verbose >= 2:
            print("\nEvaluating moves:")
            for move, prob in move_evaluations:
                if move['bluff']:
                    print(f"  Call bluff: {prob:.2%}")
                else:
                    print(f"  {move['quantity']} {move['face_value']}s: {prob:.2%}")
        
        # Select best move
        best_move, best_probability = move_evaluations[0]
        
        # Add reasoning to the move
        if best_move['bluff']:
            # Find best alternative probability if there are non-bluff moves
            alternative_moves = [(m, p) for m, p in move_evaluations if not m['bluff']]
            best_alt_prob = max(p for m, p in alternative_moves) if alternative_moves else 0.0
            best_move['reasoning'] = f"Calling bluff. probability: {best_probability:.2%}. Best alternative probability: {best_alt_prob:.2%}"
        else:
            best_move['reasoning'] = f"Move probability: {best_probability:.2%}. Bluff probability: {P_bluff:.2%}"
        
        return best_move

class AdaptiveAgent(LiarsDiceAgent):
    """Agent that adapts its strategy based on game state and player behavior patterns"""
    
    def __init__(self, name: str, color_idx: int = 0, learning_rate: float = 0.05, aggressive_factor: float = 1.0):
        super().__init__(name, color_idx)
        from liars_dice_calculator import liars_dice_calc
        self.calc_probability = liars_dice_calc
        self.player_history = {}  # {player_idx: [moves]}
        self.bluff_threshold = 0.4  # Start more aggressive at calling bluffs
        self.aggressive_factor = aggressive_factor
        self.learning_rate = learning_rate  # How quickly to adjust to player patterns
        self.predicted_bluff_rate = None  # Track predicted bluff rate
    
    def _analyze_player_patterns(self, player_idx: int) -> Dict:
        """Analyze a player's bidding patterns"""
        if player_idx not in self.player_history:
            return {'bluff_rate': 0.5, 'avg_quantity_increase': 1, 'prefers_face_increase': False}
        
        moves = self.player_history[player_idx]
        if not moves:
            return {'bluff_rate': 0.5, 'avg_quantity_increase': 1, 'prefers_face_increase': False}
        
        # Calculate actual bluff rate from outcomes
        bluffs = sum(1 for m in moves if m.get('was_bluff', False))
        bluff_rate = bluffs / len(moves) if moves else 0.5
        
        # Weight recent moves more heavily
        if len(moves) > 5:
            recent_bluffs = sum(1 for m in moves[-5:] if m.get('was_bluff', False))
            recent_rate = recent_bluffs / 5
            # Blend recent and overall rates (favor recent)
            bluff_rate = 0.7 * recent_rate + 0.3 * bluff_rate
        
        # Analyze bid patterns
        quantity_increases = []
        face_increases = []
        for i in range(1, len(moves)):
            if moves[i]['quantity'] > moves[i-1]['quantity']:
                quantity_increases.append(moves[i]['quantity'] - moves[i-1]['quantity'])
            if moves[i]['face_value'] > moves[i-1]['face_value']:
                face_increases.append(1)
        
        avg_quantity_increase = sum(quantity_increases) / len(quantity_increases) if quantity_increases else 1
        prefers_face_increase = len(face_increases) > len(quantity_increases)
        
        return {
            'bluff_rate': bluff_rate,
            'avg_quantity_increase': avg_quantity_increase,
            'prefers_face_increase': prefers_face_increase
        }
        
    def _adjust_confidence(self, game_state: Dict) -> None:
        """Adjust confidence thresholds based on game state and past performance"""
        # Get number of players and their dice
        players = game_state.get('players', [])
        if not players:
            return
        
        # More aggressive when we have more dice relative to others
        my_dice = len(self.dice)
        avg_dice = sum(p['num_dice'] for p in players) / len(players)
        dice_advantage = my_dice / avg_dice if avg_dice > 0 else 1
        
        # More aggressive when we have more lives
        my_lives = next((p['lives'] for p in players if p == game_state['current_player']), 2)
        avg_lives = sum(p['lives'] for p in players) / len(players)
        life_advantage = my_lives / avg_lives if avg_lives > 0 else 1
        
        # Calculate overall position strength
        position_strength = (dice_advantage + life_advantage) / 2
        
        # Adjust thresholds with learning rate
        target_aggression = 1.0 + (0.2 * position_strength)
        self.aggressive_factor += self.learning_rate * (target_aggression - self.aggressive_factor)
        
        # Keep aggression in reasonable range
        self.aggressive_factor = max(0.8, min(1.5, self.aggressive_factor))
    
    def _handle_bluff_call(self, was_bluff, last_move):
        """Handle the outcome of a bluff call and adjust strategy accordingly"""
        # If this was our bluff call, adjust threshold based on outcome
        if last_move.get('adjust_threshold'): # NOTE this key is only present when the bluff was called
            # Use learning rate for adjustment amount
            adjustment = self.learning_rate
            
            # If we correctly called a bluff, decrease threshold (be more aggressive)
            # If we were wrong, increase threshold (be more conservative)
            if was_bluff:
                self.bluff_threshold *= (1 - adjustment)
            else:
                self.bluff_threshold *= (1 + adjustment)

            # Make sure the threshold is within a reasonable range
            self.bluff_threshold = max(0.1, min(0.9, self.bluff_threshold))
            
            # Log the adjustment for debugging
            # print(f"Adjusted bluff threshold to {self.bluff_threshold:.2f} after {'correct' if was_bluff else 'incorrect'} bluff call")
    
    def update_from_game_state(self, game_state: Dict):
        """Update agent's strategy based on the game state"""
        super().update_from_game_state(game_state)
        
        # Check if there was a bluff call and handle it
        last_move = game_state.get('last_move', {})

        if last_move and 'was_bluff' in last_move:
            self._handle_bluff_call(last_move['was_bluff'], last_move)
    
    def make_move(self, environment) -> Dict:
        game_state = environment.get_game_state()
        last_move = game_state.get('last_move')
        valid_moves = game_state.get('valid_moves', [])
        
        # Update player history and adjust confidence
        if last_move:
            player_idx = (game_state['current_player'] - 1) % len(game_state['players'])
            if player_idx not in self.player_history:
                self.player_history[player_idx] = []
            
            # Always update history with the move
            self.player_history[player_idx].append(last_move)
        
        self._adjust_confidence(game_state)
        
        P_bluff = 0.0
        patterns = {'bluff_rate': 0.5}  # Default pattern
        
        if last_move:
            # Calculate probability of last bid being true
            face_value = last_move['face_value']
            quantity = last_move['quantity']
            my_relevant_dice = self.dice.count(face_value)
            required_successes = quantity - my_relevant_dice
            
            if required_successes <= 0:
                P_truthful = 1.0
            else:
                # Calculate probability that at least required_successes dice show face_value
                P_truthful = self.calc_probability(
                    D=game_state['total_dice'] - len(self.dice),
                    p=1/6,
                    c=required_successes,
                    k=0
                )
            
            P_bluff = 1 - P_truthful
            
            # Analyze last player's patterns
            last_player = (game_state['current_player'] - 1) % len(game_state['players'])
            patterns = self._analyze_player_patterns(last_player)
            
            # Blend mathematical probability with historical patterns
            P_bluff = 0.7 * P_bluff + 0.3 * patterns['bluff_rate']
            
            # Store predicted bluff rate for metrics
            self.predicted_bluff_rate = P_bluff
            
            # If probability of bluff is high enough, call it
            if P_bluff > self.bluff_threshold:
                move_scores = [{
                    'move': {
                        'bluff': True,
                        'predicted_bluff_rate': P_bluff,
                        'actual_bluff_rate': patterns['bluff_rate'],
                        'bluff_threshold': self.bluff_threshold,
                        'adjust_threshold': True  # Flag to indicate we should adjust threshold based on outcome
                    },
                    'score': P_bluff
                }]
                best_move = move_scores[0]
                
                # Add reasoning for debugging
                best_move['move']['reasoning'] = (
                    f"Called bluff with P(bluff)={P_bluff:.2f}, threshold={self.bluff_threshold:.2f}. "
                    f"Aggression: {self.aggressive_factor:.2f}"
                )
                
                return best_move['move']
        
        # Evaluate potential moves
        valid_moves = environment.get_valid_moves()
        move_scores = []
        
        for move in valid_moves:
            if move.get('bluff', False):
                continue  # Skip bluff moves, we already handled that case
                
            # Calculate probability of making this bid
            face_value = move['face_value']
            quantity = move['quantity']
            my_relevant_dice = self.dice.count(face_value)
            required_successes = quantity - my_relevant_dice
            
            if required_successes <= 0:
                P_success = 1.0
            else:
                P_success = self.calc_probability(
                    D=environment.get_num_dice_in_play() - len(self.dice),
                    p=1/6,
                    c=required_successes,
                    k=0
                )
            
            # Store metrics for this move
            move['predicted_bluff_rate'] = 1 - P_success
            move['actual_bluff_rate'] = patterns['bluff_rate']
            move['bluff_threshold'] = self.bluff_threshold
            
            # Score is weighted by our aggressive factor
            score = P_success * self.aggressive_factor
            move_scores.append({'move': move, 'score': score})
        
        # Select the best move
        best_move = max(move_scores, key=lambda x: x['score'])
        
        # Add reasoning for debugging
        if best_move['move'].get('bluff', False):
            best_move['move']['reasoning'] = (
                f"Called bluff with P(bluff)={P_bluff:.2f}, threshold={self.bluff_threshold:.2f}. "
                f"Aggression: {self.aggressive_factor:.2f}"
            )
        else:
            best_move['move']['reasoning'] = (
                f"Bid with score={best_move['score']:.2f}, aggression={self.aggressive_factor:.2f}. "
                f"Confidence threshold: {self.bluff_threshold:.2f}"
            )
            
        # Track learning metrics
        best_move['move']['predicted_bluff_rate'] = P_bluff
        best_move['move']['actual_bluff_rate'] = patterns['bluff_rate'] if last_move else 0.5
        best_move['move']['bluff_threshold'] = self.bluff_threshold
        
        return best_move['move']

class LLMAgent(LiarsDiceAgent):
    """Agent that uses LLM for decision making"""
    
    def __init__(self, name: str, provider: str = 'openai', model: str = 'gpt-4o-mini', 
                 api_key: str = None, color_idx: int = 0, system_prompt: str = 'naive'):    
        super().__init__(name, color_idx)
        self.provider = provider
        self.system_prompt = self._load_system_prompt(system_prompt)
        self.last_reasoning = None  # Track last reasoning
        
        if provider == 'openai':
            self.client = OpenAI(api_key=api_key)
            self.model = model or "gpt-4o-mini"
        elif provider == 'huggingface':
            self.client = InferenceClient(model=model, token=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _load_system_prompt(self, system_prompt) -> str:
        """Load the appropriate system prompt based on the specified type"""
        if system_prompt.lower() == 'naive':
            return NAIVE_SYSTEM_PROMPT
        elif system_prompt.lower() == 'informed':
            return INFORMED_SYSTEM_PROMPT
        else:
            raise ValueError(f"Unsupported system prompt type: {system_prompt}. Use 'naive' or 'informed'")
    
    def _format_game_state(self, environment) -> str:
        """Format the game state using the template from meta_prompts"""
        game_state = environment.get_game_state()
        return GAME_STATE_PROMPT.format(
            dice=self.dice,
            total_dice=game_state['total_dice'],
            last_bid=game_state.get('last_move', 'None'),
        )

    def make_move(self, environment) -> Dict:
        """Make a move using the LLM"""
        formatted_state = self._format_game_state(environment)
        
        if self.provider == 'openai':
            try:
                # Use structured output with Pydantic model
                completion = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": formatted_state}
                    ],
                    response_format=LiarsMove,
                )
                move = completion.choices[0].message.parsed
                move_json = {'quantity': move.quantity, 'face_value': move.face_value, 'bluff': move.bluff}

                # Need to check if move is valid
                game_state = environment.get_game_state()
                valid_moves = game_state.get('valid_moves', [])
                while move_json not in valid_moves:
                    # print(f"LLM Produced an Invalid move!! {move_json} retrying...")
                    invalid_prompt = INVALID_RESPONSE_PROMPT.format(
                        dice=self.dice,
                        total_dice=game_state['total_dice'],
                        last_bid=game_state.get('last_move', 'None'),
                        valid_moves=valid_moves
                    )
                    completion = self.client.beta.chat.completions.parse(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": invalid_prompt}
                        ],
                        response_format=LiarsMove,
                    )
                    move = completion.choices[0].message.parsed 
                    move_json = {'quantity': move.quantity, 'face_value': move.face_value, 'bluff': move.bluff}
                # print("LLM Produced a Valid move!!")
                # Update last reasonin
                self.last_reasoning = {
                    "decision_type": move.reasoning,
                    "move": move.dict()
                }
                return {
                    'quantity': move.quantity,
                    'face_value': move.face_value,
                    'bluff': move.bluff,
                    'reasoning': move.reasoning
                }
            except Exception as e:
                print(f"Error getting response from OpenAI: {e}")
                # Fallback to a random valid move
                game_state = environment.get_game_state()
                return random.choice(game_state.get('valid_moves', []))
        
        elif self.provider == 'huggingface':
            # Implementation for Hugging Face
            # ... (existing code)
            pass