import numpy as np

from test_utils import clingo_tensor_rule_check, tensor_rule_to_strings


# Configuration arguments
RNG_SEED = 73
NUM_NULLARY = 10
NUM_CONJUNCTS = 5
FILE_PATH = f"synth_data_in{NUM_NULLARY}_conj{NUM_CONJUNCTS}.npz"
GEN_SIZE = 10000


def evaluate_dnf(
    nullary: np.ndarray,
    and_kernel: np.ndarray,
    or_kernel: np.ndarray,
) -> np.ndarray:
    and_eval = np.min(nullary[:, None] * and_kernel + (and_kernel == 0), -1)
    or_eval = np.max(or_kernel * and_eval - (or_kernel == 0), -1)
    return or_eval


def generate_data() -> str:
    # Code from Nuri's pix2rule synthetic dataset, with minor changes

    rng = np.random.default_rng(seed=RNG_SEED)
    max_rng_tries = 100  # Number of tries before we give up generating

    # We will follow a monte carlo method and gamble with random interpretations
    # until we get enough examples. You can call it, generate and test approach
    # rather than generating a rule that we know works. Random samples are
    # easier to implement and more efficient when things scale.

    num_nullary = NUM_NULLARY
    num_conjuncts = NUM_CONJUNCTS
    targetArity = 0
    gen_size = GEN_SIZE

    in_size = NUM_NULLARY
    pred_names = [f"n{i}" for i in range(num_nullary)]

    # Generate conjunctions
    # But we want the conjuncts to be unique, here is when we start to gamble
    for i in range(max_rng_tries):
        # -1, 0, 1 is negation, not in, positive in the conjunct
        and_kernel = rng.choice([-1, 0, 1], size=(num_conjuncts, in_size))
        if np.unique(and_kernel, axis=0).shape[0] == num_conjuncts:
            # We're done, we found unique conjuncts
            break
        if i == (max_rng_tries - 1):
            raise RuntimeError(
                "Could not generate unique conjuncts, "
                "try increasing the language size."
            )

    # Now let's generate the final disjunction, luckily it's a one off
    # We only generate either in disjunction or not since having negation of
    # conjuncts does not conform to normal logic programs.
    # That is, you never get p <- not (q, r)
    or_kernel = rng.choice([0, 1], size=(num_conjuncts,))
    while not or_kernel.any():  # We want at least one conjunction
        or_kernel = rng.choice([0, 1], size=(num_conjuncts,))

    # Generate positive examples, we will reverse engineer the or_kernel and
    # and_kernel
    # If we randomly try, as we increase the language size, it becomes
    # increasingly unlikely to get a positive example.
    # So we take a more principled approach: we'll pick one conjunction to
    # satisfy
    conjunct_idx = np.flatnonzero(or_kernel)
    conjunct_idx = rng.choice(conjunct_idx, size=gen_size)
    ckernel = and_kernel[conjunct_idx]  # (B, IN)
    # Now we generate an interpretation that will satisfy ckernel
    # if the predicates are in the rule then take their value
    # otherwise it could be random, we don't care
    cmask = ckernel == 0  # (B, IN)
    pos_nullary = (
        cmask * rng.choice([-1, 1], size=(gen_size, in_size))
        + (1 - cmask) * ckernel
    )  # (B, IN)

    # Ensure all positive examples actually satisfy the kernels
    res = evaluate_dnf(pos_nullary, and_kernel, or_kernel)
    assert np.all(res == 1), "Expected positive examples to return 1"

    # Generate negative examples, it's more likely if we generate random
    # examples they will be false, so we let all the random number generator do
    # its magic
    neg_nullary = rng.choice([-1, 1], size=(gen_size, num_nullary))
    res = evaluate_dnf(neg_nullary, and_kernel, or_kernel)
    neg_idxs = np.flatnonzero(res == -1)
    neg_nullary = neg_nullary[neg_idxs]

    # Merge positive and negative examples
    nullary = np.concatenate([pos_nullary, neg_nullary], 0)
    target = np.concatenate(
        [np.ones(len(pos_nullary)), np.zeros(len(neg_nullary)) - 1], 0
    )
    data = {
        "nullary": nullary,
        "target": target,
    }

    dsize = nullary.shape[0]
    ridxs = rng.permutation(dsize)[:1000]  # (B,)
    sample = {k: v[ridxs] for k, v in data.items()}
    rule_dict = {
        "and_kernel": and_kernel,
        "or_kernel": or_kernel,
    }
    results = clingo_tensor_rule_check(sample, rule_dict)
    assert np.all(
        results == (sample["target"] == 1)
    ), "Clingo sanity check did not match target labels."

    data["and_kernel"] = and_kernel
    data["or_kernel"] = or_kernel
    data["rule_str"] = tensor_rule_to_strings(sample, rule_dict)

    # Save the file
    print(f"Creating {str(FILE_PATH)} with keys: {str(data.keys())}")
    np.savez_compressed(FILE_PATH, **data)

    # Output rules to STDOUT
    for r in rule_asp:
        print(r)

    return str(FILE_PATH)


if __name__ == "__main__":
    # Generated npz should have keys:
    # ['nullary', 'target', 'and_kernel', 'or_kernel', 'rule_str']
    generate_data()
