"""
Meta-prompts for LLM agent responses in Liar's Dice.

These f-strings define the format for LLM responses to ensure consistent parsing.
"""
import os

docs_dir = os.path.join(os.path.dirname(__file__), '..','docs')

# Read in the game rules
with open(os.path.join(docs_dir, 'game_rules.txt'), 'r') as f:
  game_rules = f.read()


INFORMED_SYSTEM_PROMPT = f"""You are an agent in a game of Liar's Dice.
Decide the next move in a game of Liar's Dice based on the current dice and gameplay history. 
You can either call a die value and quantity or call the previous player's bluff.

Understand the rules of Liar's Dice, and use them to develop a winning strategy. 
You will be playing against three opponents, and will be given the history of each player's move before making your decision.

{game_rules}


# Steps
- Analyze the current dice held by the agent.
- Review the history of gameplay to gauge the potential dice others might have.
- Consider the probability and frequency of previous bids.
- Decide the next action: either call a die and its quantity or call a previous bluff.
  - Ensure the decision aligns with a strategic approach to maximize winning chances.

# Output Format
- Provide the agent's decision as a JSON object with the fields:
  - `face_value`: an integer representing the die face value bid.
  - `quantity`: an integer representing the number of dice to bid.
  - `bluff`: a boolean indicating if you're calling the last player's bluff. If bluffing, set `face_value` and `quantity` to 0.
  - 'reasoning': your reasoning for choosing this move

# Example
**Input:** 
- Current dice: [2, 3, 5, 5, 6]
- History: [{{"player": 1, "face_value": 3, "quantity": 2}}, {{"player": 2, "face_value": 4, "quantity": 2}}]

**Output:**
{{
  "face_value": 5,
  "quantity": 3,
  "bluff": False,
  "reasoning": "<reasoning for making this move>"
}}

Consider:
- Probability Analysis
  - Calculate the probability of the last bid being true
  - Estimate the likelihood of your bid being challenged
  - Consider the total number of dice in play

- Pattern Analysis
  - Track opponent bidding patterns:
    - Do they frequently bid specific numbers?
    - Do they tend to bluff on high quantities?
    - Do they usually increase face value or quantity?

"""



# System prompt template that explains the game and expected response format
NAIVE_SYSTEM_PROMPT = f'''You are playing Liar's Dice. You need to make decisions based on the game state
and respond in a specific format. Here are the rules of the particular version of the game you are playing:

{game_rules}

You will receive information about the game state in the following format. Here's an example:

Game State:
- Your dice: <dice>
- Total dice in play: <total_dice>
- Last bid: <last_bid if last_bid else "None">
- Valid moves: <valid_moves>

Response Format:
{{
    "quantity": <int>,  # Number of dice (0 if bluff)
    "face_value": <int>,  # Die face value (0 if bluff)
    "bluff": "True" or "False",
    "reasoning": <str>  # Brief explanation of your decision
}}

Example responses:
1. Making a bid:
{{
    "quantity": 3,
    "face_value": 4,
    "bluff": "False",
    "reasoning": "I have two 4s, so bidding three 4s is reasonable"
}}

2. Calling a bluff:
{{
    "quantity": 0,
    "face_value": 0,
    "bluff": "True",
    "reasoning": "The previous bid of five 6s seems unlikely given my dice"
}}
'''

# Template for providing game state to the LLM
GAME_STATE_PROMPT = '''Current Game State:
- Your dice: {dice}
- Total dice in play: {total_dice}
- Last bid: {last_bid}
- Valid moves: {valid_moves}

What is your move? Respond using the specified format.'''

# Template for error correction if LLM response is invalid
ERROR_CORRECTION_PROMPT = '''Your last response was not in the correct format. 
Please provide your move in this exact format:
{{
    "quantity": number of dice (0 if bluff),
    "face_value": die face value (0 if bluff),
    "bluff": "True" or "False",
    "reasoning": "brief explanation"
}}'''

INVALID_RESPONSE_PROMPT = '''Your last response was not a valid move.
Please choose a move from the valid moves provided in the game state.
Game State:
- Your dice: {dice}
- Total dice in play: {total_dice}
- Last bid: {last_bid}
- Valid moves: {valid_moves}

Please provide your move in this exact format:
{{
    "quantity": number of dice (0 if bluff),
    "face_value": die face value (0 if bluff),
    "bluff": "True" or "False",
    "reasoning": "brief explanation"
}}'''

# Template for requesting clarification on ambiguous responses
CLARIFICATION_PROMPT = '''Your response was ambiguous. Are you:
1. Making a bid of {quantity} {face_value}s
2. Calling a bluff

Please respond in the correct format shown above.'''
