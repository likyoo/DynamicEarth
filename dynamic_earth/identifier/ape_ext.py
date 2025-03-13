import sys
from typing import List
import multiprocessing as mp
from detectron2.config import LazyConfig
from detectron2.data.detection_utils import read_image

sys.path.append("third_party/APE")
from third_party.APE.demo.predictor_lazy import VisualizationDemo


def build_ape(
    config_file: str,
    confidence_threshold: float = 0.5,
    opt: List[str] = [],
):
    """
    Run inference on a single input image using the specified configuration.

    Args:
        config_file (str): Path to the configuration file.
        confidence_threshold (float, optional): Minimum score for instance predictions. Defaults to 0.5.
        opt (List[str], optional): Additional command-line options for the configuration.

    Returns:
        dict: Predictions for the input image.
    """
    # load config from file and command-line arguments
    cfg = LazyConfig.load(config_file)
    cfg = LazyConfig.apply_overrides(cfg, opt)
    
    if "model_vision" in cfg.model:
        cfg.model.model_vision.test_score_thresh = confidence_threshold
    else:
        cfg.model.test_score_thresh = confidence_threshold
    
    mp.set_start_method("spawn", force=True)
    demo = VisualizationDemo(cfg)
    return demo


def extract_prediction_from_ape(
    model,
    input_path: str,
    text_prompt: str = None,
    with_box: bool = False,
    with_mask: bool = False,
    with_sseg: bool = False,
) -> dict:
    """
    Run inference on a single input image using the specified configuration.

    Args:
        input_path (str): Path to the input image.
        confidence_threshold (float, optional): Minimum score for instance predictions. Defaults to 0.5.
        text_prompt (str, optional): Text prompt for the model. Defaults to None.
        with_box (bool, optional): Whether to include bounding boxes in output. Defaults to False.
        with_mask (bool, optional): Whether to include masks in output. Defaults to False.
        with_sseg (bool, optional): Whether to include semantic segmentation in output. Defaults to False.

    Returns:
        dict: Predictions for the input image.
    """
    
    # Read and process the input image
    try:
        img = read_image(input_path, format="BGR")
    except Exception as e:
        print(f"Failed to open image: {e}")
        return {}

    predictions, visualized_output, visualized_outputs, metadata = model.run_on_image(
        img,
        text_prompt=text_prompt,
        with_box=with_box,
        with_mask=with_mask,
        with_sseg=with_sseg,
    )

    return predictions