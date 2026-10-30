import argparse
from configs.config import Config
from inference import InferenceEngine


def main():
    parser = argparse.ArgumentParser(
        description="SegFormer-B5 Footprint Refinement Predictor"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=Config.TEST_IMAGES_DIR,
        help="Path to folder containing test RGB PNG images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=Config.PREDICTIONS_DIR,
        help="Path to folder where output binary TIFFs will be saved.",
    )
    args = parser.parse_args()

    engine = InferenceEngine()
    engine.run_directory_inference(input_dir=args.input_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()