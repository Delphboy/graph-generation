import argparse
import os

import generators

_GRAPH_TYPES = {
    "semantic": generators.Semantic, 
    "spatial": generators.Sptial,
    "geometric": generators.Geometric,
    "heteroyao": generators.HeteroYao,
}


def check_args(args):
    graph_type = str.lower(args.graph_type)

    assert os.path.isdir(args.butd) and len(os.listdir(args.butd)) > 0, f"BAD DIR: The BUTD path provided ({args.butd}) is either not a directory or is empty"
    assert os.path.isdir(args.vsua) and len(os.listdir(args.vsua)) > 0, f"BAD DIR: The VSUA path provided ({args.vsua}) is either not a directory or is empty"
    assert os.path.isdir(args.sgae) and len(os.listdir(args.sgae)) > 0, f"BAD DIR: The SGAE path provided ({args.sgae}) is either not a directory or is empty"

    assert graph_type in _GRAPH_TYPES.keys(), f"BAD GRAPH: Received unsupported graph type of {args.graph_type}. Supported types are:\n\t{_GRAPH_TYPES.keys()}"

    if args.bbox != "NOT PROVIDED":
        assert os.path.isdir(args.bbox) and len(os.listdir(args.bbox)) > 0, f"BAD DIR: The bbox path provided ({args.bbox}) is either not a directory or is empty"


if __name__ == "__main__":
    args = argparse.ArgumentParser("Graph generator")

    args.add_argument("--butd", required=True, help="The directory of the bottom up top down object detection features")
    args.add_argument("--vsua", required=True, help="The directory of the data provided by the VSUA paper")
    args.add_argument("--sgae", required=True, help="The directory of the data provided by the SGAE paper")
    args.add_argument("--output_dir", required=True, help="The directory where the produced graphs will be stored once generated")
    args.add_argument("--graph_type", required=True, help=f"The graph type you wish to generate, supported graphs are: {_GRAPH_TYPES.keys()}")
    args.add_argument("--bbox", required=False, default="NOT PROVIDED", help="The directory of the bottom up top down bounding box features")

    args = args.parse_args()
    check_args(args)

    if not os.path.exists(args.output_dir):
        print(f"CREATING OUPTUT DIRECTORY: {args.output_dir}")
        os.makedirs(os.path.join(os.getcwd(), args.output_dir))

    gen = _GRAPH_TYPES[args.graph_type](args)
    gen.build_graphs()

