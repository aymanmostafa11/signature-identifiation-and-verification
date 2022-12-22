import os
from models.preprocessing import DataManager
from models.signature import SignatureClassifier, ClassicalClassifier
import matplotlib.pyplot as plt
import tensorflow as tf
from models.Pipelines import CNN_encoder_Pipeline
from PIL import Image
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", default="test", choices=["train", "test"],
                        help="Choose the mode to run the script")
    parser.add_argument("--img_path", required=True,
                        help="The path of the image to be inferred")

    args = vars(parser.parse_args())

    if args['mode'] == 'test':
        try:
            img = DataManager.read_image(args["img_path"])
        except FileNotFoundError as e:
            print(f"Img not found at given path: {args['img_path']}")
            exit()

        pipeline = CNN_encoder_Pipeline()
        pipeline.run_single(img)


