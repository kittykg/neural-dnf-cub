from typing import List

import clingo
import numpy as np

RNG_SEED = 73
NUM_NULLARY = 150
NUM_CONJUNCTS = 75
NUM_CLASSES = 25
FILE_PATH = f"synth_multiclass_data_in{NUM_NULLARY}_conj{NUM_CONJUNCTS}_out{NUM_CLASSES}.npz"
GEN_SIZE = 10000


def get_rule_asp(and_kernel: np.ndarray, or_kernel: np.ndarray) -> List[str]:
    rule_asp = []

    for i, k in enumerate(and_kernel):
        conj_atoms = []
        for j, a in enumerate(k):
            if a == 1:
                conj_atoms.append(f"a{j}")
            elif a == -1:
                conj_atoms.append(f"not a{j}")
        rule_asp.append(f"c{i} :- " + ", ".join(conj_atoms) + ".")

    for i, k in enumerate(or_kernel.T):
        for j, a in enumerate(k):
            if a == 1:
                rule_asp.append(f"class_{i} :- c{j}.")

    return rule_asp


def get_show_statements(num_labels: int) -> List[str]:
    show_statements = []
    # show_statements += [f'#show c{i}/0.' for i in range(num_conjuncts)]
    show_statements += [f"#show class_{i}/0." for i in range(num_labels)]
    return show_statements


def example_tensor_to_asp(example: np.ndarray) -> List[str]:
    return [f"a{i}." for i in range(len(example)) if example[i] == 1]


def clingo_solve(
    example_asp: List[str], rule_asp: List[str], show_statements: List[str]
) -> str:
    ctl = clingo.Control(["--warn=none"])
    ctl.add("base", [], " ".join(rule_asp + example_asp + show_statements))
    ctl.ground([("base", [])])
    with ctl.solve(yield_=True) as handle:  # type: ignore
        model = handle.model()
    return str(model)


def generate_data() -> str:
    rng = np.random.default_rng(seed=RNG_SEED)

    in_size = NUM_NULLARY
    num_conjuncts = NUM_CONJUNCTS
    num_classes = NUM_CLASSES
    gen_size = GEN_SIZE

    # For multi-class classification, only one rule is fired at once.
    # A naive way to have mutual exclusivity is that each conjunction is unique,
    # and we create an or kernel, such that a conjunction won't be used by two
    # different class, is to specify a small subset of conjunctions that a rule
    # can use, and the subsets have no intersection between each other.
    assert (
        num_conjuncts % num_classes == 0
    ), "Expected full division of NUM_CONJUNCTS / NUM_CLASSES"
    conj_to_use = int(num_conjuncts / num_classes)

    for i in range(100):
        # -1, 0, 1 is negation, not in, positive in the conjunct
        and_kernel = rng.choice([0, 1], size=(num_conjuncts, in_size))
        if np.unique(and_kernel, axis=0).shape[0] == num_conjuncts:
            # We're done, we found unique conjuncts
            break
        if i == (100 - 1):
            raise RuntimeError(
                "Could not generate unique conjuncts, "
                "try increasing the language size."
            )
    print("And kernel generated...")

    # Create or_kernel such that each rule uses a subset of conjunctions.
    or_kernel = np.zeros((num_conjuncts, num_classes)).astype(int)
    seen_sub_kernel = []
    for i in range(num_classes):
        while True:
            sub_kernel = rng.choice([0, 1], size=conj_to_use)
            if sub_kernel.any():
                # Each rule need at least one conjunction
                break
        seen_sub_kernel.append(sub_kernel)
        or_kernel[i * conj_to_use : (i + 1) * conj_to_use, i] = sub_kernel
    print("Or kernel generated...")

    rule_asp = get_rule_asp(and_kernel, or_kernel)
    show_statements = get_show_statements(num_classes)

    examples = []
    target = []
    i = gen_size

    while i > 0:
        # Pick one rule and pick one of its conjunctions
        conjunct_one_hot = or_kernel.T[rng.choice(num_classes)]
        conjunct_idx = np.where(conjunct_one_hot)[0][rng.choice(1)]
        ckernel = and_kernel[conjunct_idx]

        # Generate an example
        cmask = ckernel == 0
        example = (
            cmask * rng.choice([-1, 1], size=in_size) + (1 - cmask) * ckernel
        )

        # Check no other conjunction would be fired under this example
        multiple_classes = False
        for c in np.concatenate(
            (and_kernel[:conjunct_idx], and_kernel[conjunct_idx + 1 :]), axis=0
        ):
            idx = np.where(c != 0)[0]
            if (example[idx] == c[idx]).all():
                multiple_classes = True
                break
        if multiple_classes:
            # More than one class, ignore this example
            continue

        model = clingo_solve(
            example_tensor_to_asp(example), rule_asp, show_statements
        )
        if not model:
            # No class, ignore this example
            continue

        class_list = [int(l[6:]) for l in model.split(" ")]
        if len(class_list) > 1:
            # More than one class, ignore this example
            continue

        class_one_hot = np.zeros(num_classes).astype(int)
        class_one_hot[class_list] = 1

        examples.append(example)
        target.append(class_one_hot)
        i -= 1

    data = {
        "nullary": np.concatenate(examples).reshape((gen_size, in_size)),
        "target": np.concatenate(target).reshape((gen_size, num_classes)),
        "and_kernel": and_kernel,
        "or_kernel": or_kernel,
        "rule_str": rule_asp,
    }

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
