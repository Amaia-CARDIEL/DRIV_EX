# Copyright (c) 2026 Valeo. All rights reserved.
import os
import json
import argparse

from driv_ex import REPO_DIR


def get_args():
    parser = argparse.ArgumentParser(description="Compute counterfactual scores from algorithm results.")
    parser.add_argument(
        "--results_path", type=str, required=True,
        help=(
            "Path to the results folder to analyse. "
            "If an absolute path is given, it is used as-is. "
            "Otherwise the path is appended to REPO_DIR/results/cf_algo_results/LC_LLM/. "
            "In both cases the script reads JSON files from a best_results_json/ subfolder. "
            "Example: --results_path main_results/my_exp"
        )
    )
    parser.add_argument(
        "--dab", action="store_true",
        help="Set if results come from a DAB-based algorithm (DAB, DAB†)"
    )
    parser.add_argument(
        "--BS_lb", type=float, default=0.95,
        help="BertScore lower bound threshold for success criteria (default: 0.95)"
    )
    return parser.parse_args()


def process_mean(l, metric_name=None, BS_lb=0.95):
    if len(l) == 0:
        return 0

    if metric_name in ["yT*_has_top_rank_%", "lc_traj_coherence_%", "aggreg_and_col_%", "aggreg_score_%",
                       "%_collision_and_rank_0"] \
            or "%_BS>" in metric_name:
        return round(100 * sum([e == 1 for e in l]) / len(l), 1)

    if metric_name == "cf_driving_maneuvers":
        result = {
            "collision_%":  round(100 * sum([e == "collision"  for e in l]) / len(l), 1),
            "discomfort_%": round(100 * sum([e == "discomfort" for e in l]) / len(l), 1),
            "good_%":       round(100 * sum([e == "good"       for e in l]) / len(l), 1),
        }
        if None in l:
            result["fail_to_extract_%"] = round(100 * sum([e in [None, "fail_to_extract"] for e in l]) / len(l), 1)
        return result

    if metric_name in ["avg_script_timing", "avg_best_step"]:
        return round(sum(l) / len(l), 1)

    if metric_name == "template_filter_%":
        return round(100 * sum(v == 100 for v in l) / len(l), 1)

    if metric_name == "min_fluency":
        avg = sum(l) / len(l)
        return {"round_3": round(avg, 3), "round_4": round(avg, 4), "no_rounding": avg}

    if metric_name == "bertscore_filter_%":
        return round(100 * sum(v >= BS_lb for v in l) / len(l), 1)

    return round(sum(l) / len(l), 3)


def _empty_stats(BS_lb):
    return {
        "yT*_has_top_rank_%": [],
        "avg_p_xtg_%": [],
        "min_fluency": [],
        "template_filter_%": [],
        "lc_traj_coherence_%": [],
        "bertscore_filter_%": [],
        "aggreg_score_%": [],
        f"%_BS>{BS_lb}_and_rank_0": [],
        "%_collision_and_rank_0": [],
        "aggreg_and_col_%": [],
        "cf_driving_maneuvers": [],
        "avg_script_timing": [],
        "avg_best_step": [],
    }


def _append_collision_stats(stats, BS_lb, count_rank_0):
    """Shared end-of-file logic for collision and 3-condition metrics."""
    bs_key = f"%_BS>{BS_lb}_and_rank_0"
    collision_and_3_conds = 0

    if stats["bertscore_filter_%"][-1] >= BS_lb and count_rank_0 == 1:
        stats[bs_key].append(1)
        if stats["template_filter_%"][-1] == 100:
            stats["aggreg_score_%"].append(1)
            if stats["cf_driving_maneuvers"][-1] == "collision":
                collision_and_3_conds = 1
        else:
            stats["aggreg_score_%"].append(0)
    else:
        stats[bs_key].append(0)
        stats["aggreg_score_%"].append(0)

    stats["aggreg_and_col_%"].append(collision_and_3_conds)


def _collect_dab_file_metrics(data, stats, BS_lb):
    """Append per-file metrics from a DAB-format result to stats lists.
    Returns missing_mctp (bool)."""
    best = data["best_result"]
    missing_mctp = False

    if "best_constraint_satisfactions" in best:
        count_rank_0 = 1 if best["best_constraint_satisfactions"][0] else 0
    else:
        count_rank_0 = 1 if best["best_XTG_rank"] == 0 else 0

    stats["yT*_has_top_rank_%"].append(count_rank_0)
    stats["bertscore_filter_%"].append(best["best_sim_score"][0])
    stats["template_filter_%"].append(best["template_fitness"]["soft_eval"])
    stats["lc_traj_coherence_%"].append(best["lc_traj_coherence"])
    stats["avg_best_step"].append(best["best_steps"][0])

    mctp = best.get("min_cond_tok_prob")
    if mctp and mctp != "fail_to_extract" and "worst_prob" in mctp:
        stats["min_fluency"].append(mctp["worst_prob"][0])
    else:
        missing_mctp = True

    man_eval = best.get("man_eval_multi_sec")
    stats["cf_driving_maneuvers"].append(man_eval["4_sec"] if man_eval not in [None, "fail_to_extract"] else None)

    stats["avg_script_timing"].append(data["runtime_in_secs"])

    if man_eval not in [None, "fail_to_extract"] and man_eval["4_sec"] == "collision" and count_rank_0 == 1:
        stats["%_collision_and_rank_0"].append(1)
    else:
        stats["%_collision_and_rank_0"].append(0)

    _append_collision_stats(stats, BS_lb, count_rank_0)
    return missing_mctp


def _collect_optim_file_metrics(data, stats, BS_lb):
    """Append per-file metrics from an optim-format result (PEZ, GCG) to stats lists.
    Returns missing_mctp (bool)."""
    best = data["Best_result_wrt_prob"]
    missing_mctp = False

    count_rank_0 = 1 if best["XTG_rank"] == 0 else 0
    stats["yT*_has_top_rank_%"].append(count_rank_0)
    stats["avg_p_xtg_%"].append(best["Best_P_XTG_%"])
    stats["bertscore_filter_%"].append(best["sim_score"])
    stats["avg_best_step"].append(best["best_step"])
    stats["template_filter_%"].append(best["template_fitness"]["soft_eval"])

    mctp = best.get("min_cond_tok_prob")
    if mctp and mctp != "fail_to_extract" and "worst_prob" in mctp:
        stats["min_fluency"].append(mctp["worst_prob"][0])
    else:
        missing_mctp = True

    man_eval = best.get("maneuver_eval_multi_sec")
    if man_eval is None or man_eval == "fail_to_extract":
        stats["cf_driving_maneuvers"].append(None)
    elif "4_sec" in man_eval:
        stats["cf_driving_maneuvers"].append(man_eval["4_sec"])
    else:
        raise Exception(f"Unexpected maneuver_eval_multi_sec format: {man_eval}")

    if "lc_traj_coherence" in best:
        stats["lc_traj_coherence_%"].append(best["lc_traj_coherence"])
    else:
        print("WARNING: missing lc_traj_coherence")

    stats["avg_script_timing"].append(data["other_info"]["total_optim_runtime_in_secs"])

    if stats["cf_driving_maneuvers"][-1] == "collision" and count_rank_0 == 1:
        stats["%_collision_and_rank_0"].append(1)
    else:
        stats["%_collision_and_rank_0"].append(0)

    _append_collision_stats(stats, BS_lb, count_rank_0)
    return missing_mctp


def compute_cf_scores(results_folder, is_dab, BS_lb=0.95):
    """Read all per-sample JSON files and aggregate CF scores. Returns avg_stats dict."""
    json_folder = os.path.join(results_folder, "best_results_json")
    files = [f for f in os.listdir(json_folder) if f.endswith(".json")]

    stats = _empty_stats(BS_lb)
    missing_mctp = False

    print(f"Computing scores over {len(files)} result files.")

    for file in files:
        with open(os.path.join(json_folder, file)) as f:
            data = json.load(f)

        try:
            if is_dab:
                if data["best_result"] is None:
                    # failure case: DAB found no valid CF
                    stats["yT*_has_top_rank_%"].append(0)
                    stats["aggreg_and_col_%"].append(0)
                    stats[f"%_BS>{BS_lb}_and_rank_0"].append(0)
                    stats["aggreg_score_%"].append(0)
                    continue
                m_mctp = _collect_dab_file_metrics(data, stats, BS_lb)
            else:
                m_mctp = _collect_optim_file_metrics(data, stats, BS_lb)
        except KeyError as e:
            hint = "did you forget --dab?" if not is_dab else "did you mistakenly set --dab?"
            raise KeyError(f"{e} in {file} — {hint}") from None

        if m_mctp:
            missing_mctp = True

    if missing_mctp:
        print("WARNING: some files are missing min_fluency")

    SKIP_KEYS = {f"%_BS>{BS_lb}_and_rank_0", "%_collision_and_rank_0", "avg_p_xtg_%"}
    avg_stats = {"n_samples": len(stats["yT*_has_top_rank_%"])}
    for key, val in stats.items():
        if key not in SKIP_KEYS:
            avg_stats[key] = process_mean(val, key, BS_lb)

    return avg_stats


MAIN_METRICS = [
    "yT*_has_top_rank_%",
    "min_fluency",
    "template_filter_%",
    "bertscore_filter_%",
    "aggreg_score_%",
    "aggreg_and_col_%",
]


def print_results(avg_stats):
    print(f"\nScores for {avg_stats['n_samples']} counterfactual explanations")

    print("\n--- Main metrics ---")
    for key in MAIN_METRICS:
        if key not in avg_stats:
            continue
        val = avg_stats[key]
        if key == "min_fluency":
            print(f"{key}: {val['round_4']}")
        else:
            print(f"{key}: {val}")

    print("\n--- Complementary metrics ---")
    for key, val in avg_stats.items():
        if key in MAIN_METRICS or key == "n_samples":
            continue
        if isinstance(val, dict):
            print(f"{key}:")
            for sub_k, sub_v in val.items():
                print(f"  {sub_k}: {sub_v}")
        else:
            print(f"{key}: {val}")


def main():
    args = get_args()
    from pathlib import Path
    p = Path(args.results_path)
    results_folder = p if p.is_absolute() else REPO_DIR / "results" / "cf_algo_results" / "LC_LLM" / p

    if not os.path.isdir(results_folder):
        raise FileNotFoundError(f"Results folder not found: {results_folder}")

    print(f"Computing CF scores for: {args.results_path}")
    print(f"Algorithm type: {'DAB' if args.dab else 'optim (PEZ / GCG)'}")

    avg_stats = compute_cf_scores(str(results_folder), is_dab=args.dab, BS_lb=args.BS_lb)

    print_results(avg_stats)

    output = {
        "n_samples": avg_stats["n_samples"],
        "main_metrics":  {k: avg_stats[k] for k in MAIN_METRICS if k in avg_stats},
        "other_metrics": {k: v for k, v in avg_stats.items()
                          if k != "n_samples" and k not in MAIN_METRICS},
    }
    out_path = os.path.join(results_folder, "cf_scores.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=4)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
