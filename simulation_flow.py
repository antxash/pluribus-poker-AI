import joblib
import time
import numpy as np
from poker_ai.poker.card import Card
from poker_ai.games.short_deck.state import new_game
from poker_ai.ai.agent import Agent

def run_simulation():
    """
    This function will simulate a specific poker game scenario to determine the AI's action,
    replicating the core logic of the terminal runner and profiling performance.
    """
    start_time = time.time()
    print(f"[{time.time() - start_time:.4f}s] Script started.")

    # 1. Load card clustering data.
    try:
        card_info_lut = joblib.load('card_info_lut.joblib')
    except FileNotFoundError:
        print("Error: card_info_lut.joblib not found. Please run clustering first.")
        return
    print(f"[{time.time() - start_time:.4f}s] Loaded 'card_info_lut.joblib'.")

    # 2. Initialize the game state.
    n_players = 3
    state = new_game(n_players, card_info_lut)
    print(f"[{time.time() - start_time:.4f}s] Initialized new game state.")

    # 3. Set up the specific game scenario.
    bot_1_player = next(p for p in state.players if p.is_dealer)
    bot_2_player = next(p for p in state.players if p.is_big_blind)
    bot_2_player.cards = [Card("king", "diamonds"), Card("king", "spades")]
    bot_3_player = next(p for p in state.players if p.is_small_blind)
    
    # 4. Simulate the game history action by action.
    state = state.apply_action("raise")
    state = state.apply_action("fold")
    state = state.apply_action("call")
    flop_cards = [Card("ace", "hearts"), Card("ace", "diamonds"), Card("ace", "clubs")]
    state._table.community_cards = flop_cards
    state._betting_stage = "flop"
    state._reset_betting_round_state()
    # state = state.apply_action("raise")
    # state = state.apply_action("raise")
    print(f"[{time.time() - start_time:.4f}s] Finished simulating game history.")

    # 5. Load the pre-trained agent.
    try:
        agent = Agent("./agent/agent.joblib", use_manager=False)
    except FileNotFoundError:
        print("Error: agent.joblib not found. Please ensure a trained agent exists.")
        return
    print(f"[{time.time() - start_time:.4f}s] Loaded 'agent.joblib'. This is often the slowest step.")

    # 6. Get the recommended action using the runner's logic.
    print("\n--- AI Decision Analysis ---")
    info_set = state.info_set
    print(f"Generated InfoSet for lookup: '{info_set}'")

    # Attempt to get the strategy from the agent's dictionary
    raw_strategy = agent.strategy.get(info_set)
    final_strategy = {}

    if raw_strategy is None:
        print("\n[STRATEGY MISS] The InfoSet was not found in the agent's strategy book.")
        print("--> Creating a default strategy with uniform probabilities.")
        legal_actions = state.legal_actions
        if legal_actions:
            final_strategy = {
                action: 1 / len(legal_actions)
                for action in legal_actions
            }
    else:
        print("\n[STRATEGY HIT] Found a pre-computed strategy for this InfoSet.")
        print(f"--> Raw 'regret' values from strategy book: {raw_strategy}")
        
        # Normalize the 'regret' values into a probability distribution.
        # 1. Filter for positive regrets, as only these translate to actions.
        positive_regrets = {k: v for k, v in raw_strategy.items() if v > 0}

        if not positive_regrets:
            print("--> No positive regrets found. Defaulting to uniform strategy.")
            legal_actions = state.legal_actions
            if legal_actions:
                final_strategy = { action: 1 / len(legal_actions) for action in legal_actions }
        else:
            # 2. Calculate the sum of positive regrets.
            total_regret = sum(positive_regrets.values())
            # 3. Normalize to get probabilities.
            final_strategy = { k: v / total_regret for k, v in positive_regrets.items() }
            print("--> Processed regrets into a normalized probability distribution.")

    # Determine the recommended action (highest probability)
    recommended_action = max(final_strategy, key=final_strategy.get) if final_strategy else "No legal actions"

    print("\n--- Final AI Recommendation ---")
    print(f"Bot 2's Hand: {bot_2_player.cards}")
    print(f"Community Cards: {state.community_cards}")
    print(f"Legal Actions for Bot 2: {state.legal_actions}")
    print(f"Final Action Probabilities (Sum = {sum(final_strategy.values()):.2f}): {final_strategy}")
    print(f"AI Recommended Action for Bot 2: {recommended_action}")
    print("---------------------------")
    print(f"[{time.time() - start_time:.4f}s] Script finished.")

    print("\n--- Verification Step ---")
    try:
        # 1. Combine cards exactly like the info_set property does
        import operator
        cards = sorted(bot_2_player.cards, key=operator.attrgetter("eval_card"), reverse=True)
        cards += sorted(state.community_cards, key=operator.attrgetter("eval_card"), reverse=True)
        lookup_cards = tuple(cards)
 
        print(f"Verifying lookup for betting stage: '{state.betting_stage}'")
        print(f"Verifying lookup for combined cards: {lookup_cards}")
 
        # 2. Perform the exact lookup that happens inside the property
        cluster_id = card_info_lut[state.betting_stage][lookup_cards]
 
        print(f"[SUCCESS] Lookup successful for this stage and cards.")
        print(f"--> The returned 'cards_cluster' ID is: {cluster_id}")
        if cluster_id == 0:
            print("--> WARNING: A cluster ID of 0 is highly likely to be a default/fallback value, indicating an imprecise match.")
 
    except KeyError:
        print("[FAILURE] A KeyError occurred during manual lookup.")
        print("--> This means the combination of current stage and cards does not exist in the card_info_lut.")
    print("------------------------")

if __name__ == "__main__":
    run_simulation()
