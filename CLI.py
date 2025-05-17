#!/usr/bin/env python3
"""
CLI.py

Runs an active learning selection pipeline over a folder of images,
then pushes those selected images to Label Studio for human annotation.
"""
import argparse
import logging
import os
import glob
import pandas as pd
from omegaconf import OmegaConf

# Active learning functions
from src.active_learning import generate_pool_predictions, human_review, select_images
# Label Studio integration
from src.label_studio import get_api_key
from src.pipeline import Pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Active Learning CLI: select images and send to Label Studio"
    )
    # Image selection args
    parser.add_argument("--image_folder", required=True,
                        help="Directory containing unlabeled images.")
    parser.add_argument("--annotations_csv",
                        help="Existing preannotations CSV (skip model inference).")
    parser.add_argument("--model_path", default=None,
                        help="Detection model checkpoint (omit to load default 'tree').")
    parser.add_argument("--patch_size", type=int, default=512,
                        help="Patch size for tiling (default: 512).")
    parser.add_argument("--patch_overlap", type=float, default=0.1,
                        help="Fractional overlap between tiles (default: 0.1).")
    parser.add_argument("--min_score", type=float, default=0.1,
                        help="Min detection score to keep (default: 0.1).")
    parser.add_argument("--min_detection_score", type=float, default=0.6,
                        help="Min score for uncertain filtering (default: 0.6).")
    parser.add_argument("--confident_threshold", type=float, default=0.5,
                        help="Threshold for confident vs uncertain (default: 0.5).")
    parser.add_argument("--strategy", choices=["random","most-detections","target-labels","rarest"],
                        default="random", help="Selection strategy.")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of images to select (default: 10).")
    parser.add_argument("--pool_limit", type=int, default=1000,
                        help="Max images to subsample (default: 1000).")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for prediction (default: 16).")
    # Pool mapping
    parser.add_argument("--pool_file",
                        help="Optional CSV/TXT mapping image_path,pool_id.")
    # Label Studio args
    parser.add_argument("--ls_project_id", required=True,
                        help="Label Studio project ID to create tasks in.")
    # Outputs
    parser.add_argument("--output_images", default="selected_images.txt",
                        help="File to save selected image paths (with pool IDs).")
    parser.add_argument("--output_csv", default="selected_preannotations.csv",
                        help="File to save selected preannotations (with pool IDs).")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    # Load optional pool mapping
    pool_map = None
    if args.pool_file:
        try:
            pool_map = pd.read_csv(args.pool_file)
        except Exception:
            pool_map = pd.read_csv(args.pool_file, header=None,
                                   names=["image_path","pool_id"])
        logging.info(f"Loaded pool mapping for {len(pool_map)} images.")

    # Step 1: Load or generate preannotations
    if args.annotations_csv:
        logging.info(f"Loading preannotations from {args.annotations_csv}")
        preannotations = pd.read_csv(args.annotations_csv)
    else:
        exts = ("*.jpg","*.jpeg","*.png","*.tif","*.tiff")
        pool = []
        for ext in exts:
            pool.extend(glob.glob(os.path.join(args.image_folder, ext)))
        if not pool:
            logging.error(f"No images found in {args.image_folder}")
            return
        logging.info(f"Found {len(pool)} images — generating predictions...")

        if not args.model_path:
            from src.detection import load as load_model
            logging.info("No --model_path; loading default 'tree' model")
            model_obj = load_model("tree")
        else:
            model_obj = None

        preannotations = generate_pool_predictions(
            pool,
            patch_size=args.patch_size,
            patch_overlap=args.patch_overlap,
            min_score=args.min_score,
            model=model_obj,
            model_path=args.model_path,
            batch_size=args.batch_size,
            pool_limit=args.pool_limit
        )

    if preannotations is None or getattr(preannotations, "empty", False):
        logging.info("No predictions available; exiting.")
        return

    # Step 2: Split confident vs uncertain
    logging.info("Splitting predictions into confident vs uncertain...")
    confident, uncertain = human_review(
        preannotations,
        min_detection_score=args.min_detection_score,
        confident_threshold=args.confident_threshold
    )
    logging.info(f"Confident: {len(confident)}, Uncertain: {len(uncertain)}")

    # Step 3: Select images for annotation
    chosen, chosen_pre = select_images(
        uncertain,
        strategy=args.strategy,
        n=args.n,
        min_score=args.min_score
    )
    logging.info(f"Selected {len(chosen)} images for annotation.")

    # Merge pool IDs if available
    if pool_map is not None:
        cf = pd.DataFrame({"image_path": chosen})
        cf = cf.merge(pool_map, on="image_path", how="left")
        chosen = list(cf.itertuples(index=False, name=None))  # (image_path,pool_id)
        chosen_pre = chosen_pre.merge(pool_map, on="image_path", how="left")

    # Step 4: Save results
    with open(args.output_images, "w") as f:
        if pool_map is not None:
            for img,pid in chosen:
                f.write(f"{img},{pid}\n")
        else:
            for img in chosen:
                f.write(f"{img}\n")
    chosen_pre.to_csv(args.output_csv, index=False)
    logging.info(f"Saved images list to {args.output_images}")
    logging.info(f"Saved preannotations to {args.output_csv}")

    # Step 5: Push selected images to Label Studio
    api_key = get_api_key()
    if not api_key:
        logging.warning("No Label Studio API key found; skipping task creation.")
        return
    os.environ["LABEL_STUDIO_API_KEY"] = api_key
    logging.info("Authenticated with Label Studio.")

    # Build minimal config for legacy Pipeline
    cfg = OmegaConf.create({
        'label_studio': {'project_id': args.ls_project_id},
        'pipeline': {
            'images_to_annotate': [img for img,*_ in chosen],
            'gpus': 1
        }
    })

    pipeline = Pipeline(cfg=cfg, dask_client=None)
    pipeline.run()
    logging.info("Pushed selected images to Label Studio for annotation.")


if __name__ == "__main__":
    main()