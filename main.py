import argparse
import os

from generators import geometric, semantic, heteroyao

_GRAPH_TYPES = {
    "semantic": semantic.Semantic, 
    "geometric": geometric.Geometric,
    "heteroyao": heteroyao.HeteroYao,
}


def check_args(args):
    assert os.path.isdir(args.butd) and len(os.listdir(args.butd)) > 0, f"The BUTD path provided ({args.butd}) is either not a directory or is empty"
    assert os.path.isdir(args.vsua) and len(os.listdir(args.vsua)) > 0, f"The VSUA path provided ({args.vsua}) is either not a directory or is empty"
    assert os.path.isdir(args.sgae) and len(os.listdir(args.sgae)) > 0, f"The SGAE path provided ({args.sgae}) is either not a directory or is empty"

    # check the graph type
    assert str.lower(args.graph_type) in _GRAPH_TYPES.keys(), f"Received unsupported graph type of {args.graph_type}. Supported types are:\n\t{_GRAPH_TYPES.keys()}"


if __name__ == "__main__":
    args = argparse.ArgumentParser("Graph generator")

    args.add_argument("--butd", required=True)
    args.add_argument("--vsua", required=True)
    args.add_argument("--sgae", required=True)
    args.add_argument("--output_dir", required=True)
    args.add_argument("--graph_type", required=True)

    args = args.parse_args()
    check_args(args)

    if not os.path.exists(args.output_dir):
        print(f"CREATING OUPTUT DIRECTORY: {args.output_dir}")
        os.makedirs(os.path.join(os.getcwd(), args.output_dir))

    gen = _GRAPH_TYPES[args.graph_type](args)
    gen.build_graphs()
