import ast
from mmengine.logging import print_log
from mmdet.apis import DetInferencer
from mmdet.evaluation import get_classes


def build_mmdet_model(model, weights=None, device='cuda', palette='none'):
    """Create and return a DetInferencer instance.

    Args:
        model (str): Model configuration or checkpoint file.
        weights (str, optional): Checkpoint file. Defaults to None.
        device (str, optional): Device for inference. Defaults to 'cuda'.
        palette (str, optional): Color palette for visualization. Defaults to 'none'.

    Returns:
        DetInferencer: An instance of DetInferencer.
    """
    if weights is None and model.endswith('.pth'):
        print_log('The model is a weight file, automatically assigning to --weights')
        weights = model
        model = None

    return DetInferencer(model=model, weights=weights, device=device, palette=palette, \
        show_progress=False)


def extract_bbox_and_cls_from_mmgd(inferencer, inputs, out_dir='', texts=None, 
                                   pred_score_thr=0.3, batch_size=1, show=False, 
                                   no_save_vis=True, no_save_pred=True,
                                   print_result=False, custom_entities=True,
                                   chunked_size=-1, tokens_positive=None):
    """Perform image inference and extract bounding boxes and classes.

    Args:
        inferencer (DetInferencer): An instance of DetInferencer.
        inputs (str): Input image file or folder path.
        out_dir (str, optional): Output directory for results. Defaults to 'outputs'.
        texts (str, optional): Text prompt. Defaults to None.
        pred_score_thr (float, optional): Score threshold for predictions. Defaults to 0.3.
        batch_size (int, optional): Batch size for inference. Defaults to 1.
        show (bool, optional): Show results in a popup window. Defaults to False.
        no_save_vis (bool, optional): Do not save visualization results. Defaults to True.
        no_save_pred (bool, optional): Do not save prediction results. Defaults to True.
        print_result (bool, optional): Print the results. Defaults to False.
        custom_entities (bool, optional): Customize entity names. Defaults to True.
        chunked_size (int, optional): Size for chunked predictions. Defaults to -1.
        tokens_positive (str, optional): Positive tokens for interest areas. Defaults to None.
    """

    if no_save_vis and no_save_pred:
        out_dir = ''

    if texts is not None and texts.startswith('$:'):
        dataset_name = texts[3:].strip()
        class_names = get_classes(dataset_name)
        texts = [tuple(class_names)]

    if tokens_positive is not None:
        tokens_positive = ast.literal_eval(tokens_positive)

    # Set the chunked size
    inferencer.model.test_cfg.chunked_size = chunked_size

    call_args = {
        'inputs': inputs,
        'out_dir': out_dir,
        'texts': texts,
        'pred_score_thr': pred_score_thr,
        'batch_size': batch_size,
        'show': show,
        'no_save_vis': no_save_vis,
        'no_save_pred': no_save_pred,
        'print_result': print_result,
        'custom_entities': custom_entities,
        'tokens_positive': tokens_positive
    }

    results = inferencer(**call_args)

    return results