from typing import Dict, List

import clingo
import numpy as np
import tqdm


def run_clingo(logic_program: List[str]) -> bool:
    ctl = clingo.Control(["--warn=none"])
    ctl.add("base", [], " ".join(logic_program))
    ctl.ground([("base", [])])
    return ctl.solve().satisfiable


def tensor_rule_to_strings(
    interpretation: Dict[str, np.ndarray], rule: Dict[str, np.ndarray]
) -> List[str]:
    num_nullary = interpretation["nullary"].shape[-1]
    pred_names = [f"nullary({i})" for i in range(num_nullary)]
    conjuncts: List[str] = list()
    for conjunct in rule["and_kernel"]:
        conjs = [
            n if i == 1 else "not " + n
            for n, i in zip(pred_names, conjunct)
            if i != 0
        ]
        cstr = ", ".join(conjs)
        conjuncts.append(cstr)

    disjuncts: List[str] = list()
    head = "t"
    for i, conjunct in enumerate(conjuncts):
        if rule["or_kernel"][i] == 1:
            disjuncts.append(f"{head} :- {conjunct}.")
        elif rule["or_kernel"][i] == -1:
            propo = f"c{i}{head}"
            disjuncts.append(f"{head} :- not {propo}.")
            disjuncts.append(f"{propo} :- {conjunct}.")
    return disjuncts


def tensor_interpretations_to_strings(
    interpretation: Dict[str, np.ndarray]
) -> List[List[str]]:
    batch_size = interpretation["nullary"].shape[0]
    num_nullary = interpretation["nullary"].shape[-1]
    programs: List[List[str]] = list()
    for bidx in range(batch_size):
        nullary = interpretation["nullary"][bidx]  # (P0,)
        program = [
            f"nullary({i})." if nullary[i] == 1 else ""
            for i in range(num_nullary)
        ]
        programs.append(program)
    return programs


def clingo_tensor_rule_check(
    interpretation: Dict[str, np.ndarray],
    rule: Dict[str, np.ndarray],
    verbose: bool = True,
) -> np.ndarray:
    rule_lines: List[str] = [":- not t."] + tensor_rule_to_strings(
        interpretation, rule
    )
    results: List[bool] = list()
    interps = tensor_interpretations_to_strings(interpretation)
    for ground_interpretation in tqdm.tqdm(interps) if verbose else interps:
        results.append(run_clingo(rule_lines + ground_interpretation))
    return np.array(results)
