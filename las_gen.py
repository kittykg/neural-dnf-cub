import argparse
import pickle
import sys

from common import CUBDNDataItem


def gen_las_example(
    pkl_data_path: str, save_file_path: str, num_classes: int
) -> None:
    def gen_example_from_data(sample: CUBDNDataItem, file=sys.stdout):
        # Penalty and inclusion set
        print(
            f"#pos(eg_{sample.img_id}@{10}, "
            f"{{class({sample.label - 1})}}, {{",
            file=file,
        )

        # Exclusion set
        exclusion_set = ",\n".join(
            filter(
                lambda j: j != "",
                map(
                    lambda k: f"    class({k})"
                    if k != sample.label - 1
                    else "",
                    range(num_classes),
                ),
            )
        )
        print(exclusion_set, file=file)

        print("}, {", file=file)

        # Context
        for i, attr in enumerate(sample.attr_present_label.int()):
            if attr.item() == 0:
                continue
            print(f"    has_attr_{i}.", file=file)

        print("}).\n", file=file)

    with open(pkl_data_path, "rb") as f:
        cub_train = pickle.load(f)

    with open(save_file_path, "w") as f:
        for sample in cub_train:
            gen_example_from_data(sample, f)


def gen_las_background_knowledge(
    save_file_path: str,
    num_classes: int,
    num_attributes: int,
    is_ilasp: bool = False,
) -> None:
    with open(save_file_path, "w") as f:
        print(f"class_id(0..{num_classes - 1}).", file=f)
        print(":- class(X),  class(Y),  X < Y.", file=f)
        print("#modeh(class(const(class_id))).", file=f)
        for i in range(num_attributes):
            print(f"#modeb(has_attr_{i}).", file=f)
            if not is_ilasp:
                # FastLas requires explicit 'not' to include in hypothesis space
                print(f"#modeb(not has_attr_{i}).", file=f)
        if not is_ilasp:
            # FastLas scoring function
            print('#bias("penalty(1, head).").', file=f)
            print('#bias("penalty(1, body(X)) :- in_body(X).").', file=f)


################################################################################
#              Run below for LAS background + examples generation              #
################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-bk", type=str, help="Output file path for background")
    parser.add_argument("-e", type=str, help="Output file path for examples")
    parser.add_argument("-t", type=str, help="Input training file path")
    parser.add_argument("-nc", type=int, help="Number of classes")
    parser.add_argument("-na", type=int, help="Number of attributes")
    parser.add_argument("-ilasp", dest="is_ilasp", action="store_true")
    parser.add_argument("-fastlas", dest="is_ilasp", action="store_false")
    parser.set_defaults(is_ilasp=False)
    args = parser.parse_args()

    gen_las_background_knowledge(args.bk, args.nc, args.na, args.is_ilasp)
    gen_las_example(args.t, args.e, args.nc)
