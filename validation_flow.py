
import joblib
import operator
import random
from poker_ai.poker.card import Card
from poker_ai.games.short_deck.state import new_game, ShortDeckPokerState
from poker_ai.ai.agent import Agent

def run_validation():
    """
    This script validates the simulation approach by reverse-engineering a
    "hittable" turn scenario from the card_info_lut.joblib file.
    """
    # 1. Load card clustering data to find a valid scenario.
    print("Loading card_info_lut.joblib to find a valid turn scenario...")
    try:
        card_info_lut = joblib.load('card_info_lut.joblib')
    except FileNotFoundError:
        print("Error: card_info_lut.joblib not found. Please run clustering first.")
        return

    # 2. Extract a guaranteed "hittable" card combination from the LUT.
    if 'turn' not in card_info_lut or not card_info_lut['turn']:
        print("Error: No turn data found in card_info_lut.joblib.")
        return
    
    print(f"Keys in card_info_lut['turn']: {len(list(card_info_lut['turn'].keys()))}")
    valid_turn_key = random.choice(list(card_info_lut['turn'].keys()))
    
    if not all(isinstance(c, Card) for c in valid_turn_key):
        print("Error: The format of keys in card_info_lut.joblib is not as expected.")
        return

    hero_hand_from_lut = list(valid_turn_key[:2])
    community_cards_from_lut = list(valid_turn_key[2:])

    print(f"Found a valid scenario to test. Using hand: {hero_hand_from_lut} and community cards: {community_cards_from_lut}")

    # 3. Initialize a simple 2-player game state.
    n_players = 3
    state = new_game(n_players, card_info_lut)
    hero_player_index = 1  # Player 1 is BB (Hero) in 2-player game

    # 4. Simulate a simple game history to reach the turn.
    state._betting_stage = "pre_flop"
    state = state.apply_action("call")
    state = state.apply_action("call")

    state._betting_stage = "flop"
    state._reset_betting_round_state()
    state = state.apply_action("call")
    state = state.apply_action("call")

    state._betting_stage = "turn"
    state._reset_betting_round_state()

    # 5. Force the state to match the "hittable" scenario from the LUT.
    for p in state.players:
        p.cards = [] # Clear any random cards
    state.players[hero_player_index].cards = hero_hand_from_lut
    state._table.community_cards = community_cards_from_lut
    
    # Ensure the current player is the Hero
    while state.player_i != hero_player_index:
        state = state.apply_action("call")
    state = state.apply_action("call")

    print(f"State setup complete. It is now Player {state.player_i}'s turn (Hero).")

    # 6. Load the pre-trained agent.
    try:
        agent = Agent("./agent/agent.joblib", use_manager=False)
    except FileNotFoundError:
        print("Error: agent.joblib not found. Please ensure a trained agent exists.")
        return

    # 7. Get the recommended action for the current state.
    info_set = state.info_set
    action_probs = agent.strategy.get(info_set, state.initial_strategy)

    if not action_probs or sum(action_probs.values()) == 0:
        print("AI could not find a specific strategy for this exact situation.")
        action = f"Defaulting to legal actions: {state.legal_actions}"
    else:
        action = max(action_probs, key=action_probs.get)

    print("---")
    print(f"Hero's Hand: {state.players[hero_player_index].cards}")
    print(f"Community Cards: {state._table.community_cards}")
    print(f"Legal Actions: {state.legal_actions}")
    print(f"AI Recommended Action for Hero: {action}")
    print("---")

if __name__ == "__main__":
    run_validation()
