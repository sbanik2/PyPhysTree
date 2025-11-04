from math import log
import numpy as np
from collections import deque
import logging
from typing import Callable, List, Optional, Tuple, Deque,Dict,Any
from collections import deque
from random import random
from .utils import Perturbate




class Node:
    """
    A class representing a single node in the MCTS tree.
    """
    def __init__(self, node_id: int, parent: Optional['Node'] = None, depth: int = 0):
        self.node_id = node_id
        self.parent = parent
        self.depth = depth
        self.visits = 1
        self.children: List[int] = []
        self.playout_data: List[int] = []
        self.score: Optional[float] = None
        self.parameter: Optional[List[float]] = None

    def add_child(self, child_id: int) -> None:
        """Add a child node by its ID."""
        self.children.append(child_id)

    def add_playout(self, idx: int) -> None:
        """Add a playout index."""
        self.playout_data.append(idx)

    def update_score(self, score: float, parameter: List[float]) -> None:
        """Update the node's score and associated parameter."""
        self.score = score
        self.parameter = parameter






class MCTS:
    """
    The Continuous Monte Carlo Tree Search (MCTS) algorithm.
    """

    def __init__(
        self,
        root_data: List[float],
        evaluate: Callable[[List[float]], Tuple[List[float], float]],
        nplayouts: int = 10,
        explore_constant_max: float = 1,
        explore_constant_min: float = 0.1,
        max_depth: int = 12,
        a: float = 3,
        logger: Optional[logging.Logger] = None
    ) -> None:
        """
        Initialize the MCTS class.

        Args:
        - root_data: The initial data for the root node.
        - perturbate: The perturbation function used to generate new parameters.
        - evaluate: The evaluation function used to score the parameters.
        - nplayouts: The number of playouts per node expansion.
        - explore_constant_max: The maximum value of the exploration constant.
        - explore_constant_min: The minimum value of the exploration constant.
        - max_depth: The maximum depth of the tree.
        - a: A scaling factor for exploration.
        - logger: A logger instance to log the details of MCTS operations.
        """
        
        # Initialization of MCTS
        self.evaluate = evaluate
        self.nplayouts = nplayouts
        self.max_depth = max_depth
        self.explore_constant_max = explore_constant_max
        self.explore_constant_min = explore_constant_min
        self.explore_constant = explore_constant_max
        
        # Logger setup
        self.logger = logger or logging.getLogger(__name__)

        
        # Initialize MCTS with the root node
        self.idx: int = 0
        self.root_node = Node(node_id=0)
        data_relaxed, score = self.evaluate(root_data)
        self.root_node.update_score(score, data_relaxed)
        self.score_list: List[float] = [score]
        self.parameter_list: List[List[float]] = [data_relaxed]
        self.perform_playouts(self.root_node, self.nplayouts)
        self.nodes: Dict[int, Node] = {0: self.root_node}

    def perform_playouts(self, node: Node, nplayouts: int) -> None:
        """
        Perform playouts for the given node.

        Args:
        - node: The node to perform playouts for.
        - nplayouts: The number of playouts to perform.
        """
        for _ in range(nplayouts):
            self.idx += 1
            playdata = self.perturbate(node.parameter, depth=node.depth)
            playdata_relaxed, playscore = self.evaluate(playdata)
            node.add_playout(self.idx)
            self.score_list.append(playscore)
            self.parameter_list.append(playdata_relaxed)


    def expand_node(self, parent_node: Node) -> None:
        """
        Perform expansion for a given parent node and return the updated state.

        Args:
        - parent_node: The parent node to expand.
        """
        parent_node.visits += 1
        best_play_index = min(parent_node.playout_data, key=lambda idx: self.score_list[idx])
        best_play_score = self.score_list[best_play_index]

        # Create a new child node
        child_node = Node(node_id=self.idx+1, parent=parent_node, depth=parent_node.depth + 1)
        self.nodes[child_node.node_id] = child_node
        child_node.update_score(best_play_score, self.parameter_list[best_play_index])

        parent_node.add_child(child_node.node_id)
        self.perform_playouts(child_node, self.nplayouts)

    def simulate_node(self, node: Node) -> None:
        """
        Perform simulation for a given node and update its state.

        Args:
        - node: The node to simulate.
        """
        node.visits += 1
        self.perform_playouts(node, self.nplayouts)

    def get_descendants(self, node: Node) -> List[int]:
        """
        Get all descendants' IDs of a node (all child nodes and their children recursively).

        Args:
        - node: The node whose descendants are to be found.

        Returns:
        - descendants_ids: A list of all descendant node IDs.
        """
        descendants_ids: List[int] = []
        nodes_to_process: Deque[Node] = deque([node])  # Start with the current node

        while nodes_to_process:
            current_node = nodes_to_process.popleft()
            descendants_ids.append(current_node.node_id)

            nodes_to_process.extend(self.nodes[child_id] for child_id in current_node.children)

        return descendants_ids

    def backpropagate_and_select(self) -> Optional[Node]:
        """
        Perform backpropagation and node selection based on UCB, considering the descendants' minimum score.
        If the depth of a node is greater than max_depth, assign it a very high negative score.

        Returns:
        - best_node: The node with the highest UCB score.
        """
        best_node = None
        best_score = -float('inf')

        for node in self.nodes.values():
            if node.depth > self.max_depth:
                ucb_score = -1e10  # Assign a very high negative score to penalize deep nodes
            else:
                if node.parent is None:
                    continue  # Skip the root node

                descendants_ids = self.get_descendants(node)
                all_indexes = []
                for descendant_id in descendants_ids:
                    descendant_node = self.nodes[descendant_id]
                    all_indexes.extend(descendant_node.playout_data)

                all_indexes.extend(node.playout_data)
                best_reward = min(self.score_list[i] for i in all_indexes)
                ucb_score = -best_reward + self.explore_constant * np.sqrt(
                    log(self.nodes[node.parent.node_id].visits) / node.visits
                )

            if ucb_score > best_score:
                best_score = ucb_score
                best_node = node

        return best_node

        

    def run(
        self,
        niterations: int = 200,
        nexpand: int = 3,
        nsimulate: int = 1,
        patience: int = 10,
        tolerance: float = 1e-6,
        score_threshold_iterations: int = 5,
        head_data: Optional[List[List[float]]] = None
    ) -> Tuple[Dict[int, Node], List[float], List[List[float]]]:
        """
        Run the MCTS for the specified number of iterations and expansions.
        The loop will terminate early if the best score does not change within `patience` iterations.
        Additionally, the exploration constant will be adjusted based on the score change.

        Args:
        - patience: The number of iterations to wait before stopping if the best score doesn't change.
        - tolerance: The precision level to consider if the score has stopped improving (default: 1e-6).
        - score_threshold_iterations: The number of iterations to check for score change before adjusting the exploration constant.
        - head_data: Head data for initial expansion (optional).

        Returns:
        - nodes: The nodes in the MCTS tree.
        - score_list: The list of scores corresponding to each playout.
        - parameter_list: The list of parameters corresponding to each playout.
        """
        self.perform_playouts(self.root_node, self.nplayouts)


        self.expand_node(self.root_node)

        best_score = min(self.score_list)
        patience_counter = 0
        no_improvement_counter = 0

        for iteration in range(niterations):
            for _ in range(nexpand):
                selected_node = self.backpropagate_and_select()
                self.expand_node(selected_node)

            for _ in range(nsimulate):
                selected_node = self.backpropagate_and_select()
                self.simulate_node(selected_node)

            current_best_score = min(self.score_list)

            if round(current_best_score, int(-np.log(tolerance))) == round(best_score, int(-np.log(tolerance))):
                patience_counter += 1
                no_improvement_counter += 1
            else:
                patience_counter = 0
                no_improvement_counter = 0

            best_score = current_best_score

            if no_improvement_counter >= score_threshold_iterations:
                self.explore_constant = self.explore_constant_max
                self.logger.info(f"Score hasn't improved for {score_threshold_iterations} iterations. Setting explore_constant to max.")
            else:
                self.explore_constant = self.explore_constant_min
                self.logger.info(f"Exploration constant set to minimum due to score decrease.")

            self.logger.info(f"Iteration {iteration + 1}: Total evaluations: {len(self.score_list)}, Best score so far: {best_score}, Exploration constant: {self.explore_constant}")

            if patience_counter >= patience:
                self.logger.info(f"Early stopping after {patience_counter} iterations with no improvement.")
                break

        return self.nodes, self.score_list, self.parameter_list





