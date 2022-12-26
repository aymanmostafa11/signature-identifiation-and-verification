import os
from models.Pipelines import CNN_encoder_Pipeline
import argparse


root_dir = os.path.abspath(".")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["infer", "evaluate"], help="Choose the mode to run the"
                                                                                   "pipeline on")
    parser.add_argument("-p", "--path", help="the path to infer/evaluate"
                                             "if 'train' or 'test' is provided "
                                             "the process is done on the existing data", required=True)

    args = parser.parse_args()

    if args.mode == "infer":
        path = os.path.join(root_dir, args.path)
        if os.path.isdir(path):  # batch mode
            pass  # TODO: Run pipeline on batch
        else:
            try:
                pipeline = CNN_encoder_Pipeline()
                pipeline.run_single(path)
            except FileNotFoundError as e:
                print(f"Img not found at given path: {args['img_path']}")
                exit()

    elif args.mode == "evaluate":

        if args.path in ["train", "test"]:  # predefined data
            pipeline = CNN_encoder_Pipeline()
            pipeline.evaluate(args.path)
        else:  # new data
            pass
