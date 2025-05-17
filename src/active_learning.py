import random
from src import detection
import pandas as pd
import geopandas as gpd

def human_review(
    predictions: pd.DataFrame,
    min_detection_score: float = 0.6,
    min_classification_score: float = 0.5,
    confident_threshold: float = 0.5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Predict on images and divide into confident and uncertain predictions.
    If no classification („cropmodel_score“) is present, fall back to detection‐only split.
    """
    # fallback if no classification scores were generated
    if "cropmodel_score" not in predictions.columns:
        confident = predictions[predictions["score"] >= confident_threshold]
        uncertain = predictions[predictions["score"] <  confident_threshold]
        return confident, uncertain

    # otherwise use full detection + classification logic
    filtered = predictions[
        (predictions["score"] >= min_detection_score) &
        (predictions["cropmodel_score"] < min_classification_score)
    ]

    uncertain = filtered[filtered["cropmodel_score"] <= confident_threshold]
    confident = filtered[
        ~filtered["image_path"].isin(uncertain["image_path"])
    ]
    return confident, uncertain


def generate_pool_predictions(
    pool: list[str],
    patch_size: int = 512,
    patch_overlap: float = 0.1,
    min_score: float = 0.1,
    model=None,
    model_path: str = None,
    dask_client=None,
    batch_size: int = 16,
    pool_limit: int = 1000,
    crop_model=None
) -> pd.DataFrame | None:
    """
    Generate predictions for the flight pool.
    """
    if len(pool) > pool_limit:
        pool = random.sample(pool, pool_limit)

    preannotations = detection.predict(
        m=model,
        model_path=model_path,
        image_paths=pool,
        patch_size=patch_size,
        patch_overlap=patch_overlap,
        batch_size=batch_size,
        crop_model=crop_model,
        dask_client=dask_client
    )

    if not preannotations:
        return None

    preannotations = pd.concat(preannotations)
    if preannotations.empty:
        return None

    preannotations = gpd.GeoDataFrame(preannotations, geometry="geometry")
    return preannotations[preannotations["score"] >= min_score]


def select_images(
    preannotations: pd.DataFrame,
    strategy: str,
    n: int = 10,
    target_labels: list[str] = None,
    min_score: float = 0.3
) -> tuple[list[str], pd.DataFrame | None]:
    """
    Select images to annotate based on the strategy.
    """
    if preannotations.empty:
        return [], None

    if strategy == "random":
        chosen = random.sample(preannotations["image_path"].unique().tolist(), n)
    else:
        pre = preannotations[preannotations["score"] >= min_score]
        if strategy == "most-detections":
            chosen = (
                pre.groupby("image_path")
                   .size()
                   .sort_values(ascending=False)
                   .head(n)
                   .index
                   .tolist()
            )
        elif strategy == "target-labels":
            if not target_labels:
                raise ValueError("Target labels are required for the 'target-labels' strategy.")
            chosen = (
                pre[pre["cropmodel_label"].isin(target_labels)]
                .groupby("image_path")["score"]
                .mean()
                .sort_values(ascending=False)
                .head(n)
                .index
                .tolist()
            )
        elif strategy == "rarest":
            counts = pre.groupby("cropmodel_label").size().sort_values()
            pre["label_count"] = pre["cropmodel_label"].map(counts)
            pre = pre.sort_values("label_count", ascending=True)
            chosen = pre.drop_duplicates("image_path").head(n)["image_path"].tolist()
        else:
            raise ValueError(f"Invalid strategy '{strategy}'")

    chosen, preannotations[preannotations["image_path"].isin(chosen)]



    # Example usage:
    
    # python CLI.py `
    # >>   --image_folder .\Vulture_03_28_2022 `
    # >>   --patch_size 512 `
    # >>   --patch_overlap 0.1 `
    # >>   --min_score 0.1 `
    # >>   --min_detection_score 0.6 `
    # >>   --confident_threshold 0.5 `
    # >>   --strategy most-detections `
    # >>   --n 2 `
    # >>   --batch_size 16 `
    # >>   --pool_limit 5 `
    # >>   --ls_project_id 42 `
    # >>   --output_images random_detected_images.txt `
    # >>   --output_csv random_preannotations.csv
