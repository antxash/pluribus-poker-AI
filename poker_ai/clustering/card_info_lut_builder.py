import logging
import time
from pathlib import Path
from typing import Any, Dict, List
import concurrent.futures

import joblib
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import wasserstein_distance
from tqdm import tqdm

from poker_ai.clustering.card_combos import CardCombos
from poker_ai.clustering.game_utility import GameUtility
from poker_ai.clustering.preflop import compute_preflop_lossless_abstraction

log = logging.getLogger("poker_ai.clustering.runner")


class CardInfoLutBuilder(CardCombos):
    """
    Stores info buckets for each street when called

    Attributes
    ----------
    card_info_lut : Dict[str, Any]
        Lookup table of card combinations per betting round to a cluster id.
    centroids : Dict[str, Any]
        Centroids per betting round for use in clustering previous rounds by
        earth movers distance.
    """

    def __init__(
        self,
        n_simulations_river: int,
        n_simulations_turn: int,
        n_simulations_flop: int,
        low_card_rank: int,
        high_card_rank: int,
        save_dir: str,
    ):
        self.n_simulations_river = n_simulations_river
        self.n_simulations_turn = n_simulations_turn
        self.n_simulations_flop = n_simulations_flop
        self.low_card_rank = low_card_rank
        self.high_card_rank = high_card_rank
        super().__init__(
            low_card_rank, high_card_rank,
        )
        self.save_dir = Path(save_dir)
        self.card_info_lut_path: Path = self.save_dir / "card_info_lut.joblib"
        self.centroid_path: Path = self.save_dir / "centroids.joblib"
        try:
            self.card_info_lut: Dict[str, Any] = joblib.load(self.card_info_lut_path)
            self.centroids: Dict[str, Any] = joblib.load(self.centroid_path)
        except FileNotFoundError:
            self.centroids: Dict[str, Any] = {}
            self.card_info_lut: Dict[str, Any] = {}

    def compute(
        self, n_river_clusters: int, n_turn_clusters: int, n_flop_clusters: int,
    ):
        """Compute all clusters and save to card_info_lut dictionary.

        Will attempt to load previous progress and will save after each cluster
        is computed.
        """
        log.info("Starting computation of clusters.")
        start = time.time()
        if "pre_flop" not in self.card_info_lut:
            self.card_info_lut["pre_flop"] = compute_preflop_lossless_abstraction(
                builder=self
            )
            joblib.dump(self.card_info_lut, self.card_info_lut_path)
        
        if "river" not in self.card_info_lut:
            self.card_info_lut["river"] = self._compute_river_clusters(
                n_river_clusters, n_turn_clusters, n_flop_clusters
            )
            joblib.dump(self.card_info_lut, self.card_info_lut_path)
            joblib.dump(self.centroids, self.centroid_path)
        if "turn" not in self.card_info_lut:
            self.card_info_lut["turn"] = self._compute_turn_clusters(
                n_river_clusters, n_turn_clusters, n_flop_clusters
            )
            joblib.dump(self.card_info_lut, self.card_info_lut_path)
            joblib.dump(self.centroids, self.centroid_path)
        if "flop" not in self.card_info_lut:
            self.card_info_lut["flop"] = self._compute_flop_clusters(
                n_river_clusters, n_turn_clusters, n_flop_clusters
            )
            joblib.dump(self.card_info_lut, self.card_info_lut_path)
            joblib.dump(self.centroids, self.centroid_path)
        end = time.time()
        log.info(f"Finished computation of clusters - took {end - start} seconds.")

    def _get_full_params(self, n_river_clusters: int, n_turn_clusters: int, n_flop_clusters: int) -> Dict:
        """Helper to create a dictionary of all parameters for checkpoint validation."""
        return {
            'low_card_rank': self.low_card_rank,
            'high_card_rank': self.high_card_rank,
            'n_simulations_river': self.n_simulations_river,
            'n_simulations_turn': self.n_simulations_turn,
            'n_simulations_flop': self.n_simulations_flop,
            'n_river_clusters': n_river_clusters,
            'n_turn_clusters': n_turn_clusters,
            'n_flop_clusters': n_flop_clusters
        }

    def _compute_river_clusters(self, n_river_clusters: int, n_turn_clusters: int, n_flop_clusters: int):
        """Compute river clusters and create lookup table."""
        log.info("Starting computation of river clusters.")
        start_time = time.time()
        
        checkpoint_path = self.save_dir / "river_ehs_checkpoint.joblib"
        current_params = self._get_full_params(n_river_clusters, n_turn_clusters, n_flop_clusters)

        start_batch = 0
        all_river_ehs = []
        try:
            checkpoint_data = joblib.load(checkpoint_path)
            if checkpoint_data.get('params') == current_params:
                start_batch = checkpoint_data['completed_batches'] + 1
                all_river_ehs = checkpoint_data['results']
                log.info(f"Resumed from checkpoint. Last completed batch: {start_batch - 1}. Starting from batch {start_batch + 1}.")
            else:
                log.warning("Checkpoint parameters do not match current parameters. Discarding old checkpoint and starting from scratch.")
        except FileNotFoundError:
            log.info("No checkpoint found for river computation. Starting from scratch.")
        except Exception as e:
            log.warning(f"Could not load checkpoint file due to an error: {e}. Starting from scratch.")

        total_combinations = len(self.river)
        batch_size = 10000
        total_batches = (total_combinations + batch_size - 1) // batch_size
        log.info(f"Processing {total_combinations:,} river combinations in {total_batches:,} batches of {batch_size:,} each.")

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for batch_idx in range(start_batch, total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, total_combinations)
                batch_combinations = self.river[start_idx:end_idx]
                
                log.info(f"Processing batch {batch_idx + 1}/{total_batches} (combinations {start_idx:,} to {end_idx:,})")
                
                batch_ehs = list(
                    tqdm(
                        executor.map(
                            self.process_river_ehs,
                            batch_combinations,
                            chunksize=max(1, len(batch_combinations) // 160),
                        ),
                        total=len(batch_combinations),
                        desc=f"Batch {batch_idx + 1}/{total_batches}",
                        unit="combinations"
                    )
                )
                
                all_river_ehs.extend(batch_ehs)
                
                checkpoint_data = {'params': current_params, 'completed_batches': batch_idx, 'results': all_river_ehs}
                joblib.dump(checkpoint_data, checkpoint_path)
                log.info(f"Completed batch {batch_idx + 1}/{total_batches}. Checkpoint saved. Total combinations processed: {len(all_river_ehs):,}")

        log.info("All river batches completed. Starting final clustering.")
        self._river_ehs = all_river_ehs
        
        self.centroids["river"], self._river_clusters = self.cluster(
            num_clusters=n_river_clusters, X=self._river_ehs
        )
        
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            log.info(f"Removed river checkpoint file: {checkpoint_path}")

        end_time = time.time()
        log.info(f"Finished computation of river clusters - took {end_time - start_time:.2f} seconds.")
        return self.create_card_lookup(self._river_clusters, self.river)

    def _compute_turn_clusters(self, n_river_clusters: int, n_turn_clusters: int, n_flop_clusters: int):
        """Compute turn clusters and create lookup table."""
        log.info("Starting computation of turn clusters.")
        start_time = time.time()

        checkpoint_path = self.save_dir / "turn_ehs_checkpoint.joblib"
        current_params = self._get_full_params(n_river_clusters, n_turn_clusters, n_flop_clusters)

        start_batch = 0
        all_turn_ehs_distributions = []
        try:
            checkpoint_data = joblib.load(checkpoint_path)
            if checkpoint_data.get('params') == current_params:
                start_batch = checkpoint_data['completed_batches'] + 1
                all_turn_ehs_distributions = checkpoint_data['results']
                log.info(f"Resumed from checkpoint. Last completed batch: {start_batch - 1}. Starting from batch {start_batch + 1}.")
            else:
                log.warning("Checkpoint parameters do not match current parameters. Discarding old checkpoint and starting from scratch.")
        except FileNotFoundError:
            log.info("No checkpoint found for turn computation. Starting from scratch.")
        except Exception as e:
            log.warning(f"Could not load checkpoint file due to an error: {e}. Starting from scratch.")

        total_combinations = len(self.turn)
        batch_size = 10000
        total_batches = (total_combinations + batch_size - 1) // batch_size
        log.info(f"Processing {total_combinations:,} turn combinations in {total_batches:,} batches of {batch_size:,} each.")

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for batch_idx in range(start_batch, total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, total_combinations)
                batch_combinations = self.turn[start_idx:end_idx]
                
                log.info(f"Processing batch {batch_idx + 1}/{total_batches} (combinations {start_idx:,} to {end_idx:,})")
                
                batch_ehs_distributions = list(
                    tqdm(
                        executor.map(
                            self.process_turn_ehs_distributions,
                            batch_combinations,
                            chunksize=max(1, len(batch_combinations) // 160),
                        ),
                        total=len(batch_combinations),
                        desc=f"Batch {batch_idx + 1}/{total_batches}",
                        unit="combinations"
                    )
                )
                
                all_turn_ehs_distributions.extend(batch_ehs_distributions)
                
                checkpoint_data = {'params': current_params, 'completed_batches': batch_idx, 'results': all_turn_ehs_distributions}
                joblib.dump(checkpoint_data, checkpoint_path)
                log.info(f"Completed batch {batch_idx + 1}/{total_batches}. Checkpoint saved. Total combinations processed: {len(all_turn_ehs_distributions):,}")

        log.info("All turn batches completed. Starting final clustering.")
        self._turn_ehs_distributions = all_turn_ehs_distributions
        
        self.centroids["turn"], self._turn_clusters = self.cluster(
            num_clusters=n_turn_clusters, X=self._turn_ehs_distributions
        )
        
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            log.info(f"Removed turn checkpoint file: {checkpoint_path}")

        end_time = time.time()
        log.info(f"Finished computation of turn clusters - took {end_time - start_time:.2f} seconds.")
        return self.create_card_lookup(self._turn_clusters, self.turn)

    def _compute_flop_clusters(self, n_river_clusters: int, n_turn_clusters: int, n_flop_clusters: int):
        """Compute flop clusters and create lookup table."""
        log.info("Starting computation of flop clusters.")
        start_time = time.time()

        checkpoint_path = self.save_dir / "flop_ehs_checkpoint.joblib"
        current_params = self._get_full_params(n_river_clusters, n_turn_clusters, n_flop_clusters)

        start_batch = 0
        all_flop_potential_aware_distributions = []
        try:
            checkpoint_data = joblib.load(checkpoint_path)
            if checkpoint_data.get('params') == current_params:
                start_batch = checkpoint_data['completed_batches'] + 1
                all_flop_potential_aware_distributions = checkpoint_data['results']
                log.info(f"Resumed from checkpoint. Last completed batch: {start_batch - 1}. Starting from batch {start_batch + 1}.")
            else:
                log.warning("Checkpoint parameters do not match current parameters. Discarding old checkpoint and starting from scratch.")
        except FileNotFoundError:
            log.info("No checkpoint found for flop computation. Starting from scratch.")
        except Exception as e:
            log.warning(f"Could not load checkpoint file due to an error: {e}. Starting from scratch.")

        total_combinations = len(self.flop)
        batch_size = 10000
        total_batches = (total_combinations + batch_size - 1) // batch_size
        log.info(f"Processing {total_combinations:,} flop combinations in {total_batches:,} batches of {batch_size:,} each.")

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for batch_idx in range(start_batch, total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, total_combinations)
                batch_combinations = self.flop[start_idx:end_idx]
                
                log.info(f"Processing batch {batch_idx + 1}/{total_batches} (combinations {start_idx:,} to {end_idx:,})")
                
                batch_potential_aware_distributions = list(
                    tqdm(
                        executor.map(
                            self.process_flop_potential_aware_distributions,
                            batch_combinations,
                            chunksize=max(1, len(batch_combinations) // 160),
                        ),
                        total=len(batch_combinations),
                        desc=f"Batch {batch_idx + 1}/{total_batches}",
                        unit="combinations"
                    )
                )
                
                all_flop_potential_aware_distributions.extend(batch_potential_aware_distributions)
                
                checkpoint_data = {'params': current_params, 'completed_batches': batch_idx, 'results': all_flop_potential_aware_distributions}
                joblib.dump(checkpoint_data, checkpoint_path)
                log.info(f"Completed batch {batch_idx + 1}/{total_batches}. Checkpoint saved. Total combinations processed: {len(all_flop_potential_aware_distributions):,}")

        log.info("All flop batches completed. Starting final clustering.")
        self._flop_potential_aware_distributions = all_flop_potential_aware_distributions
        
        self.centroids["flop"], self._flop_clusters = self.cluster(
            num_clusters=n_flop_clusters, X=self._flop_potential_aware_distributions
        )
        
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            log.info(f"Removed flop checkpoint file: {checkpoint_path}")

        end_time = time.time()
        log.info(f"Finished computation of flop clusters - took {end_time - start_time:.2f} seconds.")
        return self.create_card_lookup(self._flop_clusters, self.flop)

    def simulate_get_ehs(self, game: GameUtility,) -> np.ndarray:
        """
        Get expected hand strength object.

        Parameters
        ----------
        game : GameUtility
            GameState for help with determining winner and sampling opponent hand

        Returns
        -------
        ehs : np.ndarray
            [win_rate, loss_rate, tie_rate]
        """
        ehs: np.ndarray = np.zeros(3)
        for _ in range(self.n_simulations_river):
            idx: int = game.get_winner()
            # increment win rate for winner/tie
            ehs[idx] += 1 / self.n_simulations_river
        return ehs

    def simulate_get_turn_ehs_distributions(
        self,
        available_cards: np.ndarray,
        the_board: np.ndarray,
        our_hand: np.ndarray,
    ) -> np.ndarray:
        """
        Get histogram of frequencies that a given turn situation resulted in a
        certain cluster id after a river simulation.

        Parameters
        ----------
        available_cards : np.ndarray
            Array of available cards on the turn
        the_board : np.nearray
            The board as of the turn
        our_hand : np.ndarray
            Cards our hand (Card)

        Returns
        -------
        turn_ehs_distribution : np.ndarray
            Array of counts for each cluster the turn fell into by the river
            after simulations
        """
        turn_ehs_distribution = np.zeros(len(self.centroids["river"]))
        # sample river cards and run a simulation
        for _ in range(self.n_simulations_turn):
            river_card = np.random.choice(available_cards, 1, replace=False)
            board = np.append(the_board, river_card)
            game = GameUtility(our_hand=our_hand, board=board, cards=self._cards)
            ehs = self.simulate_get_ehs(game)
            # get EMD for expected hand strength against each river centroid
            # to which does it belong?
            for idx, river_centroid in enumerate(self.centroids["river"]):
                emd = wasserstein_distance(ehs, river_centroid)
                if idx == 0:
                    min_idx = idx
                    min_emd = emd
                else:
                    if emd < min_emd:
                        min_idx = idx
                        min_emd = emd
            # now increment the cluster to which it belongs -
            turn_ehs_distribution[min_idx] += 1 / self.n_simulations_turn
        return turn_ehs_distribution

    def process_river_ehs(self, public: np.ndarray) -> np.ndarray:
        """
        Get the expected hand strength for a particular card combo.

        Parameters
        ----------
        public : np.ndarray
            Cards to process

        Returns
        -------
            Expected hand strength
        """
        our_hand = public[:2]
        board = public[2:7]
        # Get expected hand strength
        game = GameUtility(our_hand=our_hand, board=board, cards=self._cards)
        return self.simulate_get_ehs(game)

    @staticmethod
    def get_available_cards(
        cards: np.ndarray, unavailable_cards: np.ndarray
    ) -> np.ndarray:
        """
        Get all cards that are available.

        Parameters
        ----------
        cards : np.ndarray
        unavailable_cards : np.array
            Cards that are not available.

        Returns
        -------
            Available cards
        """
        # Turn into set for O(1) lookup speed.
        unavailable_cards = set(unavailable_cards.tolist())
        return np.array([c for c in cards if c not in unavailable_cards])

    def process_turn_ehs_distributions(self, public: np.ndarray) -> np.ndarray:
        """
        Get the potential aware turn distribution for a particular card combo.

        Parameters
        ----------
        public : np.ndarray
            Cards to process

        Returns
        -------
            Potential aware turn distributions
        """
        available_cards: np.ndarray = self.get_available_cards(
            cards=self._cards, unavailable_cards=public
        )
        # sample river cards and run a simulation
        turn_ehs_distribution = self.simulate_get_turn_ehs_distributions(
            available_cards, the_board=public[2:6], our_hand=public[:2],
        )
        return turn_ehs_distribution

    def process_flop_potential_aware_distributions(
        self, public: np.ndarray,
    ) -> np.ndarray:
        """
        Get the potential aware flop distribution for a particular card combo.

        Parameters
        ----------
        public : np.ndarray
            Cards to process

        Returns
        -------
            Potential aware flop distributions
        """
        available_cards: np.ndarray = self.get_available_cards(
            cards=self._cards, unavailable_cards=public
        )
        potential_aware_distribution_flop = np.zeros(len(self.centroids["turn"]))
        for j in range(self.n_simulations_flop):
            # randomly generating turn
            turn_card = np.random.choice(available_cards, 1, replace=False)
            our_hand = public[:2]
            board = public[2:5]
            the_board = np.append(board, turn_card).tolist()
            # getting available cards
            available_cards_turn = np.array(
                [x for x in available_cards if x != turn_card[0]]
            )
            turn_ehs_distribution = self.simulate_get_turn_ehs_distributions(
                available_cards_turn, the_board=the_board, our_hand=our_hand,
            )
            for idx, turn_centroid in enumerate(self.centroids["turn"]):
                # earth mover distance
                emd = wasserstein_distance(turn_ehs_distribution, turn_centroid)
                if idx == 0:
                    min_idx = idx
                    min_emd = emd
                else:
                    if emd < min_emd:
                        min_idx = idx
                        min_emd = emd
            # Now increment the cluster to which it belongs.
            potential_aware_distribution_flop[min_idx] += 1 / self.n_simulations_flop
        return potential_aware_distribution_flop

    @staticmethod
    def cluster(num_clusters: int, X: np.ndarray):
        km = KMeans(
            n_clusters=num_clusters,
            init="random",
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0,
        )
        y_km = km.fit_predict(X)
        # Centers to be used for r - 1 (ie; the previous round)
        centroids = km.cluster_centers_
        return centroids, y_km

    @staticmethod
    def create_card_lookup(clusters: np.ndarray, card_combos: np.ndarray) -> Dict:
        """
        Create lookup table.

        Parameters
        ----------
        clusters : np.ndarray
            Array of cluster ids.
        card_combos : np.ndarray
            The card combos to which the cluster ids belong.

        Returns
        -------
        lossy_lookup : Dict
            Lookup table for finding cluster ids.
        """
        log.info("Creating lookup table.")
        lossy_lookup = {}
        for i, card_combo in enumerate(tqdm(card_combos)):
            lossy_lookup[tuple(card_combo)] = clusters[i]
        return lossy_lookup
