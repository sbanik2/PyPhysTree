from typing import Callable, List, Optional, Tuple, Dict, Any
import numpy as np
import logging
from math import log
from collections import deque
from random import random
from .MCTS import MCTS
from .utils import _latin, adaptive_bounds, Perturbate
import random







class MCTSBatch:
    def __init__(
        self,
        objfunc: Any,
        config_dict: Dict[float, Dict[str, List[float]]],
        logger: logging.Logger,
        nplayouts: int = 5,
        explore_constant_max: float = 1e20,
        explore_constant_min: float = 1e-20,
        max_depth: int = 12,
        niterations: int = 5,
        nexpand: int = 2,
        nsimulate: int = 1,
        patience: int = 3,
        tolerance: float = 1e-1,
        score_threshold_iterations: int = 2,
        verbose: bool = True
    ):
        self.objfunc = objfunc
        self.config_dict = config_dict
        self.logger = logger

        # MCTS parameters
        self.nplayouts = nplayouts
        self.explore_constant_max = explore_constant_max
        self.explore_constant_min = explore_constant_min
        self.max_depth = max_depth
        self.niterations = niterations
        self.nexpand = nexpand
        self.nsimulate = nsimulate
        self.patience = patience
        self.tolerance = tolerance
        self.score_threshold_iterations = score_threshold_iterations
        self.verbose = verbose

        self.best_result: Optional[Tuple[float, List[float]]] = None


    def run(self) -> Tuple[Dict[float, Dict[str, Any]], int]:

        best_stats = {}
        total_calculations: int = 0
        best_result = None
        
        for i, config in self.config_dict.items():
            root_data = config["root_data"]
            head_data = config["head_data"]
            lb = config["lower_bounds"]
            ub = config["upper_bounds"]
            a = config["a"]

            self.logger.info(f"Running MCTS with a = {a}")
            self.logger.info(f"Root data: {root_data}")
            self.logger.info(f"Head data: {head_data}")
            self.logger.info(f"Bounds: lower = {lb}, upper = {ub}")

            perturbator = Perturbate(lb, ub, a=a, b=1 / 2 ** 0.5)

            mcts = MCTS(
                root_data=root_data,
                perturbate=perturbator.generate_new_parameter,
                evaluate=self.objfunc,
                nplayouts=self.nplayouts,
                explore_constant_max=self.explore_constant_max,
                explore_constant_min=self.explore_constant_min,
                max_depth=self.max_depth,
                logger=self.logger
            )

            nodes, score_list, parameter_list = mcts.run(
                niterations=self.niterations,
                nexpand=self.nexpand,
                nsimulate=self.nsimulate,
                patience=self.patience,
                tolerance=self.tolerance,
                score_threshold_iterations=self.score_threshold_iterations,
                head_data=head_data
            )

            minscore, mindepth, minidx = min(
                (
                    (min(scores), node.depth, node.playout_data[scores.index(min(scores))])
                    for node in nodes.values()
                    if (scores := [score_list[i] for i in node.playout_data])
                ),
                default=(1e300, None, None)
            )

            best_stats[i] = {
                "a": a,
                "score": minscore,
                "depth": mindepth,
                "parameter": parameter_list[minidx],
                "lb": lb,
                "ub": ub
            }

            if best_result is None or minscore < best_result[0]:
                best_result = (minscore, parameter_list[minidx])

            total_calculations += len(score_list)
            self.logger.info(f"Best score for a = {a}: {minscore}")
            self.logger.info(f"Best parameter found for a = {a}: {parameter_list[minidx]}")

        if self.verbose:
            print(f"Best score from MCTS ensemble: {best_result[0]}")
            print(f"Best parameter found: {best_result[1]}")

        return best_stats, total_calculations







def run_global_tree_batch(global_batch: MCTSBatch, top_k: int = 10):
    stats_global, total_evals = global_batch.run()

    sorted_stats = sorted(stats_global.items(), key=lambda x: x[1]["score"])[:top_k]
    #top_stats = select_lowest_score_from_clusters(stats_global,k=top_k) #{idx: result for idx, result in sorted_stats}
    top_stats = {idx: result for idx, result in sorted_stats}


    return top_stats, total_evals




def run_local_tree_batch(
    local_batch: MCTSBatch,
    top_stats: Dict[int, Dict],
    objfunc,
    logger,
    a_min: float = 0.0008,
    a_max: float = 0.08,
    n_iterations: int = 10,
    patience_threshold: int = 2,
    loop_threshold: int = 4,
    scale_step: float = 0.005
):
    prev_best = {idx: dict(result) for idx, result in top_stats.items()}
    scales = {idx: 1.0 for idx in top_stats}
    stagnation_counter = {idx: 0 for idx in top_stats}
    total_evals_local = 0

    minscore, best_parameter = min(
        ((v['score'], v['parameter']) for v in top_stats.values()),
        key=lambda x: x[0]  # Compare only by score
    )
    loop_stagnation_counter = 0

    for i in range(n_iterations):
        logger.info(f"=== Local Optimization Round {i + 1} ===")

        new_config_dict = {}
        improved = False

        for idx, result in top_stats.items():
            if stagnation_counter[idx] > patience_threshold:
                logger.info(f"Pruned config id={idx} due to stagnation.")
                continue

            curr_score = result["score"]
            curr_param = np.array(result["parameter"])
            curr_a = result["a"]
            curr_depth = result["depth"]
            curr_lb = np.array(result["lb"])
            curr_ub = np.array(result["ub"])
            prev_score = prev_best[idx]["score"]

            if curr_score < prev_score:
                scales[idx] *= (1 + scale_step)
                stagnation_counter[idx] = 0
                prev_best[idx] = dict(result)
                ref_param = curr_param
                ref_a = np.clip(curr_a, a_min, a_max)
            else:
                scales[idx] *= (1 - scale_step)
                stagnation_counter[idx] += 1
                ref = prev_best[idx]
                ref_param = np.array(ref["parameter"])
                ref_a = np.clip(ref["a"], a_min, a_max)
                curr_lb = np.array(ref["lb"])
                curr_ub = np.array(ref["ub"])
                curr_depth = ref["depth"]

            base_scale = Perturbate(curr_lb, curr_ub, a=ref_a, b=1 / np.sqrt(2)).DepthScale(curr_depth + 1)
            scale = base_scale * scales[idx]

            new_ub, new_lb = adaptive_bounds(
                ub=curr_ub,
                lb=curr_lb,
                scale=scale,
                params=ref_param
            )

            new_config_dict[idx] = {
                "a": ref_a * scales[idx],
                "root_data": ref_param,
                "head_data": None,
                "lower_bounds": new_lb,
                "upper_bounds": new_ub
            }

        if not new_config_dict:
            logger.info(f"All configs pruned at iteration {i + 1}. Exiting.")
            break

        # Run updated local batch
        local_batch.config_dict = new_config_dict
        stats_local, evals = local_batch.run()
        total_evals_local += evals

        for idx in new_config_dict:
            if idx in stats_local:
                top_stats[idx] = stats_local[idx]

        # Evaluate new best after local update
        round_best_score = minscore
        for key in top_stats:
            score = top_stats[key]['score']
            if score < minscore:
                minscore = score
                best_parameter = top_stats[key]['parameter']
                improved = True

        if not improved:
            loop_stagnation_counter += 1
        else:
            loop_stagnation_counter = 0

        print(f"\nAfter local tree iteration {i + 1}:")
        print(f"  Best score       : {minscore}")
        print(f"  Best parameter   : {best_parameter}")
        logger.info(f"Round {i + 1} best score: {minscore} with parameter: {best_parameter}")

        if loop_stagnation_counter > loop_threshold:
            logger.info("Early stopping due to lack of improvement.")
            break

    return top_stats, total_evals_local, minscore, best_parameter







def run_optimisation_mcts(
    objfunc,
    logger,
    lb: np.ndarray,
    ub: np.ndarray,
    ntrees: int,
    top_k: int,
    a_min=0.0008,
    a_max=0.08,
    explore_constant_min: float = 1e1,
    niterations: int = 5,
    nexpand: int = 2,
    nsimulate: int = 1,
    nplayouts: int = 5,
    max_depth_global: int = 12,
    n_local_loop: int = 10,
    patience_threshold_pruning: int = 2,
    patience_threshold_local_loop: int = 3,
    scale_step_factor: float = 0.005,
    seed_value: int = 1234
):
    random.seed(seed_value)
    np.random.seed(seed_value)

    a_values = np.linspace(a_min, a_max, ntrees)

    # -------------------- Generate Initial Root Points --------------------
    head_points = _latin(ntrees, lb, ub)

    config_dict = {
        i: {
            "a": a,
            "root_data": head_points[i],
            "head_data": None,
            "lower_bounds": lb,
            "upper_bounds": ub
        }
        for i, a in enumerate(a_values)
    }

    # -------------------- Construct Global Batch --------------------
    global_batch = MCTSBatch(
        objfunc=objfunc,
        config_dict=config_dict,
        logger=logger,
        explore_constant_min=explore_constant_min,
        explore_constant_max=1e20,
        max_depth=max_depth_global,
        niterations=niterations,
        nexpand=nexpand,
        nsimulate=nsimulate,
        nplayouts=nplayouts,
        patience=2,
        tolerance=1e-2,
        score_threshold_iterations=1,
        verbose=True
    )

    print("========== Global Trees run commenced =============")
    logger.info("Starting global MCTS batch...")

    # -------------------- Run Global Search --------------------
    top_stats, total_evals_global = run_global_tree_batch(global_batch, top_k=top_k)

    # -------------------- Prepare Local Config --------------------
    local_config_dict = {
        idx: {
            "a": result["a"],
            "root_data": result["parameter"],
            "head_data": None,
            "lower_bounds": result["lb"],
            "upper_bounds": result["ub"]
        }
        for idx, result in top_stats.items()
    }

    # -------------------- Construct Local Batch --------------------
    local_batch = MCTSBatch(
        objfunc=objfunc,
        config_dict=local_config_dict,
        logger=logger,
        max_depth=int(1e6),
        niterations=niterations,
        nexpand=nexpand,
        nsimulate=nsimulate,
        patience=1,
        nplayouts=nplayouts,
        tolerance=1e-20,
        score_threshold_iterations=int(1e6),
        explore_constant_min=1e-40,
        explore_constant_max=1e-40,
        verbose=False
    )

    print("\n\n========== Local Trees run commenced =============")
    logger.info("Starting local MCTS refinement...")

    # -------------------- Run Local Refinement --------------------
    top_stats_refined, total_evals_local, minscore, best_parameter = run_local_tree_batch(
        local_batch=local_batch,
        top_stats=top_stats,
        objfunc=objfunc,
        logger=logger,
        a_min=a_min,
        a_max=a_max,
        n_iterations=n_local_loop,
        patience_threshold=patience_threshold_pruning,
        loop_threshold = patience_threshold_local_loop,
        scale_step=scale_step_factor
        
    )

    # -------------------- Final Output --------------------
    total_evals = total_evals_global + total_evals_local
    logger.info(f"Total evaluations performed: {total_evals}")
    print(f"========== Optimization Complete ==========")
    print(f"Total evaluations: {total_evals}")

    return minscore, best_parameter





