# dds cloudapi for Grounding DINO 1.5
from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk import Client
from dds_cloudapi_sdk import DetectionTask
from dds_cloudapi_sdk import TextPrompt
from dds_cloudapi_sdk import DetectionModel
from dds_cloudapi_sdk import DetectionTarget

import os
import cv2
import torch
import tempfile
import numpy as np
import supervision as sv


def extract_bbox_and_cls_from_dino1_5(
    token,
    img_path,
    text_prompt='house',
    grounding_model=DetectionModel.GDino1_5_Pro,
    box_threshold=0.2,
    with_slice_inference=False,
    slice_wh=(480, 480),
    overlap_ratio=(0.2, 0.2),
    ):

    """
    Prompt Grounding DINO 1.5 with Text for Box Prompt Generation with Cloud API
    """
    # Step 1: initialize the config
    config = Config(token)

    # Step 2: initialize the client
    client = Client(config)

    # Step 3: run the task by DetectionTask class
    # image_url = "https://algosplt.oss-cn-shenzhen.aliyuncs.com/test_files/tasks/detection/iron_man.jpg"
    # if you are processing local image file, upload them to DDS server to get the image url

    classes = [x.strip().lower() for x in text_prompt.split('.') if x]
    class_name_to_id = {name: id for id, name in enumerate(classes)}
    class_id_to_name = {id: name for name, id in class_name_to_id.items()}

    if with_slice_inference:
        def callback(image_slice: np.ndarray) -> sv.Detections:
            print("Inference on image slice")
            # save the img as temp img file for GD-1.5 API usage
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmpfile:
                temp_filename = tmpfile.name
            cv2.imwrite(temp_filename, image_slice)
            image_url = client.upload_file(temp_filename)
            task = DetectionTask(
                image_url=image_url,
                prompts=[TextPrompt(text=text_prompt)],
                targets=[DetectionTarget.BBox],  # detect bbox
                model=grounding_model,  # detect with GroundingDino-1.5-Pro model
                bbox_threshold=box_threshold, # box confidence threshold
            )
            client.run_task(task)
            result = task.result
            # detele the tempfile
            os.remove(temp_filename)
            
            input_boxes = []
            confidences = []
            class_ids = []
            objects = result.objects
            for idx, obj in enumerate(objects):
                input_boxes.append(obj.bbox)
                confidences.append(obj.score)
                cls_name = obj.category.lower().strip()
                class_ids.append(class_name_to_id[cls_name])
            # ensure input_boxes with shape (_, 4)
            input_boxes = np.array(input_boxes).reshape(-1, 4)
            class_ids = np.array(class_ids)
            confidences = np.array(confidences)
            return sv.Detections(xyxy=input_boxes, confidence=confidences, class_id=class_ids)
        
        slicer = sv.InferenceSlicer(
            callback=callback,
            slice_wh=slice_wh,
            overlap_ratio_wh=overlap_ratio,
            iou_threshold=0.5,
            overlap_filter_strategy=sv.OverlapFilter.NON_MAX_SUPPRESSION
            )
        detections = slicer(cv2.imread(img_path))
        class_names = [class_id_to_name[id] for id in detections.class_id]
        confidences = detections.confidence
        class_ids = detections.class_id
        input_boxes = detections.xyxy
    else:
        image_url = client.upload_file(img_path)

        task = DetectionTask(
            image_url=image_url,
            prompts=[TextPrompt(text=text_prompt)],
            targets=[DetectionTarget.BBox],  # detect bbox
            model=grounding_model,  # detect with GroundingDINO-1.5-Pro model
            bbox_threshold=box_threshold, # box confidence threshold
        )

        client.run_task(task)
        result = task.result

        objects = result.objects  # the list of detected objects

        input_boxes = []
        confidences = []
        class_names = []
        class_ids = []

        for idx, obj in enumerate(objects):
            input_boxes.append(obj.bbox)
            confidences.append(obj.score)
            cls_name = obj.category.lower().strip()
            class_names.append(cls_name)
            class_ids.append(class_name_to_id[cls_name])

        input_boxes = np.array(input_boxes)
        class_ids = np.array(class_ids)

    return input_boxes, class_ids