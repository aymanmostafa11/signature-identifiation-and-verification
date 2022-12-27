import os
from models.Pipelines import CNN_encoder_Pipeline
import argparse


root_dir = os.path.abspath(".")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["infer", "evaluate"], help="Choose the mode to run the"
                                                                                   "pipeline on")
    parser.add_argument("-p", "--path", help="the path to infer/evaluate\n"
                                             "if 'train' or 'test' is provided"
                                             "the process is done on the existing data", required=True)

    args = parser.parse_args()

    pipeline = CNN_encoder_Pipeline()
    if args.mode == "infer":
        abs_path = os.path.join(root_dir, args.path)
        if os.path.isdir(abs_path):  # batch mode
            pipeline.run_bulk(abs_path)
        else:
            pipeline.run_single(abs_path)

    elif args.mode == "evaluate":
        if args.path in ["train", "test"]:  # predefined data
            pipeline.evaluate(args.path)
        else:  # new data
            pass
