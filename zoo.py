import logging
import os
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np
import torch
from PIL import Image
import json

import fiftyone as fo
from fiftyone import Model, SamplesMixin
from fiftyone.core.labels import Detection, Detections, Keypoint, Keypoints, Classification, Classifications, Polyline, Polylines

from transformers import AutoProcessor, Gemma3ForConditionalGeneration

# DEFAULT_DETECTION_SYSTEM_PROMPT = """You are a helpful assistant. You specialize in detection and localization of any meaningful visual elements. Please detect both primary elements and their associated components when relevant to the instruction.  
# Use descriptive, context-specific labels that include the parent object name when labeling parts. Report bbox coordinates and label for each object as a JSON array of predictions in the format: {"bbox_2d": [y_min, x_min, y_max, x_max], "label": "label"}. Note that coordinates are ordered as [y_min, x_min, y_max, x_max], not the typical [x_min, y_min, x_max, y_max]. Say nothing else."""

# DEFAULT_KEYPOINT_SYSTEM_PROMPT = """You are a helpful assistant. You specialize in key point detection across any visual domain. A key point represents the center of any meaningful visual element. 

# Key points should adapt to the context (physical world, digital interfaces, artwork, etc.) while maintaining consistent accuracy and relevance. Use descriptive, context-specific labels that include the parent object name when labeling parts.

# Report all key points as a JSON array of predictions in the format: {"point_2d": [y, x], "label": "description"}. Note that coordinates are ordered as [y, x], not the typical [x, y]. Say nothing else."""

DEFAULT_CLASSIFICATION_SYSTEM_PROMPT = """You are a helpful assistant. You specialize in comprehensive classification across any visual domain.

Unless specifically requested for single-class output, multiple relevant classifications can be provided. Report all classifications as JSON array of predictions in the format: [{"label": "label"}]. Say nothing else."""

DEFAULT_VQA_SYSTEM_PROMPT = "You are a helpful assistant. You provide clear and concise answers to questions about images. Report answers in natural language text in English."

# DEFAULT_SEGMENTATION_PROMPT="""You are a helpful assistant. You specialize in generating segmentation masks as polylines. 

# For each object, provide a list of contours where each contour is a list of interleaved y,x coordinates in pixel space. Report all contours as a JSON array of predictions in the format: {"label": "label", "point_2d": [[y1,x1,y2,x2,...], [y3,x3,y4,x4,...], ...]} where each inner list represents one complete contour. Note that coordinates are ordered as [y, x], not the typical [x, y]. Say nothing else."""

GEMMA_OPERATIONS = {
    "vqa": {
        "system_prompt": DEFAULT_VQA_SYSTEM_PROMPT,
        "params": {},
    },
    # model can't do these tasks yet
    # "segment": {
    #     "system_prompt": DEFAULT_SEGMENTATION_PROMPT,
    #     "params": {},
    # },
    # "detect": {
    #     "system_prompt": DEFAULT_DETECTION_SYSTEM_PROMPT,
    #     "params": {},
    # },
    # "point": {
    #     "system_prompt": DEFAULT_KEYPOINT_SYSTEM_PROMPT,
    #     "params": {},
    # },
    "classify": {
        "system_prompt": DEFAULT_CLASSIFICATION_SYSTEM_PROMPT,
        "params": {},
    }
}

# MODEL_SIZE = 896 # gemma resizes images to 896 x 896

logger = logging.getLogger(__name__)

# Utility functions
def get_device():
    """Get the appropriate device for model inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

class Gemma3(SamplesMixin, Model):
    """A FiftyOne model for running Gemma3 vision tasks"""

    def __init__(
        self,
        model_path: str,
        operation: str = None,
        prompt: str = None,
        system_prompt: str = None,
        **kwargs
    ):
        if operation not in GEMMA_OPERATIONS:
            raise ValueError(f"Invalid operation: {operation}. Must be one of {list(GEMMA_OPERATIONS.keys())}")
        
        self._fields = {}
        
        self.model_path = model_path
        self._custom_system_prompt = system_prompt  # Store custom system prompt if provided
        self._operation = operation
        self.prompt = prompt
        
        self.device = get_device()
        logger.info(f"Using device: {self.device}")

        # Set dtype for CUDA devices
        self.torch_dtype = torch.bfloat16 if self.device == "cuda" else None
        # Load model and processor
        logger.info(f"Loading model from {model_path}")

        if self.torch_dtype:
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                model_path,
                trust_remote_code=True,
                # local_files_only=True,
                device_map=self.device,
                torch_dtype=self.torch_dtype
            )
        else:
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                model_path,
                trust_remote_code=True,
                # local_files_only=True,
                device_map=self.device,
            )
        
        logger.info("Loading processor")
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            use_fast=True
        )

    def _get_field(self):
        if "prompt_field" in self.needs_fields:
            prompt_field = self.needs_fields["prompt_field"]
        else:
            prompt_field = next(iter(self.needs_fields.values()), None)

        return prompt_field

    @property
    def media_type(self):
        return "image"
    
    @property
    def operation(self):
        return self._operation

    @operation.setter
    def operation(self, value):
        if value not in GEMMA_OPERATIONS:
            raise ValueError(f"Invalid operation: {value}. Must be one of {list(GEMMA_OPERATIONS.keys())}")
        self._operation = value

    @property
    def system_prompt(self):
        # Return custom system prompt if set, otherwise return default for current operation
        return self._custom_system_prompt if self._custom_system_prompt is not None else GEMMA_OPERATIONS[self.operation]["system_prompt"]

    @system_prompt.setter
    def system_prompt(self, value):
        self._custom_system_prompt = value

    def _parse_json(self, s: str) -> Optional[Dict]:
        """Parse JSON from model output.
        
        The model may return JSON in different formats:
        1. Raw JSON string
        2. JSON wrapped in markdown code blocks (```json ... ```)
        3. Non-JSON string (returns None)
        
        Args:
            s: String output from the model to parse
            
        Returns:
            Dict: Parsed JSON dictionary if successful
            None: If parsing fails or input is invalid
            Original input: If input is not a string
        """
        # Return input directly if not a string
        if not isinstance(s, str):
            return s
            
        # Handle JSON wrapped in markdown code blocks
        if "```json" in s:
            try:
                # Extract JSON between ```json and ``` markers
                s = s.split("```json")[1].split("```")[0].strip()
            except:
                pass
        
        # Attempt to parse the JSON string
        try:
            return json.loads(s)
        except:
            # Log first 200 chars of failed parse for debugging
            logger.debug(f"Failed to parse JSON: {s[:200]}")
            return None

    # def _to_detections(self, boxes: List[Dict], image_width: int, image_height: int) -> fo.Detections:
    #     """Convert bounding boxes to FiftyOne Detections.
        
    #     Takes a list of bounding box dictionaries and converts them to FiftyOne Detection 
    #     objects with normalized coordinates. Handles both single boxes and lists of boxes,
    #     including boxes nested in dictionaries.

    #     Args:
    #         boxes: List of dictionaries or single dictionary containing bounding box info.
    #             Each box should have:
    #             - 'bbox_2d' or 'bbox': List of [x1,y1,x2,y2] coordinates in pixel space
    #             - 'label': Optional string label (defaults to "object")
    #         image_width: Width of the image in pixels
    #         image_height: Height of the image in pixels

    #     Returns:
    #         fo.Detections object containing the converted bounding box annotations
    #         with coordinates normalized to [0,1] x [0,1] range
    #     """
    #     detections = []
        
    #     # Calculate scaling factors
    #     scale_x = image_width / MODEL_SIZE
    #     scale_y = image_height / MODEL_SIZE
        
    #     # Handle case where boxes is a dictionary
    #     if isinstance(boxes, dict):
    #         boxes = next((v for v in boxes.values() if isinstance(v, list)), boxes)
        
    #     boxes = boxes if isinstance(boxes, list) else [boxes]
        
    #     for box in boxes:
    #         try:
    #             bbox = box.get('bbox_2d', box.get('bbox', None))
    #             if not bbox:
    #                 continue
                
    #             # Correctly interpret [y_min, x_min, y_max, x_max] format
    #             y_min, x_min, y_max, x_max = map(float, bbox)
                
    #             # Scale from model space to original image space
    #             x_min = x_min * scale_x
    #             x_max = x_max * scale_x
    #             y_min = y_min * scale_y
    #             y_max = y_max * scale_y
                
    #             # Normalize to [0,1] for FiftyOne
    #             x = x_min / image_width
    #             y = y_min / image_height
    #             w = (x_max - x_min) / image_width
    #             h = (y_max - y_min) / image_height
                
    #             detection = fo.Detection(
    #                 label=str(box.get("label", "object")),
    #                 bounding_box=[x, y, w, h],
    #             )
    #             detections.append(detection)
                
    #         except Exception as e:
    #             logger.debug(f"Error processing box {box}: {e}")
    #             continue
                
    #     return fo.Detections(detections=detections)

    # def _to_keypoints(self, points: List[Dict], image_width: int, image_height: int) -> fo.Keypoints:
    #     """Convert a list of point dictionaries to FiftyOne Keypoints.
        
    #     Args:
    #         points: List of dictionaries containing point information.
    #             Each point should have:
    #             - 'point_2d': List of [x,y] coordinates in pixel space
    #             - 'label': String label describing the point
    #         image_width: Width of the image in pixels
    #         image_height: Height of the image in pixels
                
    #     Returns:
    #         fo.Keypoints object containing the converted keypoint annotations
    #         with coordinates normalized to [0,1] x [0,1] range
        
    #     Expected input format:
    #     [
    #         {"point_2d": [100, 200], "label": "person's head", "confidence": 0.9},
    #         {"point_2d": [300, 400], "label": "dog's nose"}
    #     ]
    #     """
    #     keypoints = []
        
    #     # Calculate scaling factors
    #     scale_x = image_width / MODEL_SIZE
    #     scale_y = image_height / MODEL_SIZE

    #     for point in points:
    #         try:
    #             # Points are returned as [y, x] instead of [x, y]
    #             y, x = point["point_2d"]
    #             x = float(x.cpu() if torch.is_tensor(x) else x)
    #             y = float(y.cpu() if torch.is_tensor(y) else y)
                
    #             # Scale from model space to original image space
    #             x = x * scale_x
    #             y = y * scale_y

    #             normalized_point = [
    #                 x / image_width,
    #                 y / image_height
    #             ]

    #             keypoint = fo.Keypoint(
    #                 label=str(point.get("label", "point")),
    #                 points=[normalized_point],
    #             )
    #             keypoints.append(keypoint)

    #         except Exception as e:
    #             logger.debug(f"Error processing point {point}: {e}")
    #             continue

    #     return fo.Keypoints(keypoints=keypoints)
    
    # def _to_polylines(self, predictions: List[Dict], image_width: int, image_height: int) -> fo.Polylines:
    #     """Convert model predictions to FiftyOne Polylines format.
        
    #     Args:
    #         predictions: List of dictionaries containing polyline information.
    #             Each prediction should have:
    #             - 'label': String label for the object
    #             - 'polylines': List of contours where each contour is a list of 
    #             interleaved [x1,y1,x2,y2,...] coordinates in pixel space
    #         image_width: Width of the image in pixels
    #         image_height: Height of the image in pixels
            
    #     Returns:
    #         fo.Polylines object containing the converted polyline annotations
    #         with coordinates normalized to [0,1] x [0,1] range
    #     """
    #     polylines = []
        
    #     # Calculate scaling factors
    #     scale_x = image_width / MODEL_SIZE
    #     scale_y = image_height / MODEL_SIZE
        
    #     for pred in predictions:
    #         try:              
    #             label = str(pred.get("label", "object"))
    #             contours = []
                
    #             for contour in pred["point_2d"]:
    #                 # Convert to normalized [0,1] coordinates
    #                 points = []
    #                 # For polylines, the coordinates are interleaved [y1, x1, y2, x2, ...]
    #                 for i in range(0, len(contour), 2):
    #                     if i+1 < len(contour):
    #                         # Get coordinates in the right order
    #                         y = float(contour[i])
    #                         x = float(contour[i+1])
                            
    #                         # Scale from model space to original image space
    #                         x = x * scale_x
    #                         y = y * scale_y
                            
    #                         # Normalize to [0,1]
    #                         x = x / image_width
    #                         y = y / image_height
    #                         points.append([x, y])
                            
    #                 contours.append(points)

    #             polyline = fo.Polyline(
    #                 label=label,
    #                 points=contours,
    #                 filled=True,
    #                 closed=True,
    #             )
    #             polylines.append(polyline)
                
    #         except Exception as e:
    #             logger.debug(f"Error processing prediction {pred}: {e}")
    #             continue

    #     return fo.Polylines(polylines=polylines)

    def _to_classifications(self, classes: List[Dict]) -> fo.Classifications:
        """Convert a list of classification dictionaries to FiftyOne Classifications.
        
        Args:
            classes: List of dictionaries containing classification information.
                Each dictionary should have:
                - 'label': String class label
                
        Returns:
            fo.Classifications object containing the converted classification 
            annotations with labels and optional confidence scores
            
        Example input:
            [
                {"label": "cat",},
                {"label": "dog"}
            ]
        """
        classifications = []

        # Process each classification dictionary
        for cls in classes:
            try:
                # Create Classification object with required label and optional confidence
                classification = fo.Classification(
                    label=str(cls["label"]),  # Convert label to string for consistency
                )
                classifications.append(classification)

            except Exception as e:
                # Log any errors but continue processing remaining classifications
                logger.debug(f"Error processing classification {cls}: {e}")
                continue

        # Return Classifications container with all processed results
        return fo.Classifications(classifications=classifications)

    def _predict(self, image: Image.Image, sample=None) -> Union[fo.Detections, fo.Keypoints, fo.Classifications, fo.Polylines, str]:
        """Process a single image through the model and return predictions."""
        if sample is not None and self._get_field() is not None:
            field_value = sample.get_field(self._get_field())
            if field_value is not None:
                self.prompt = str(field_value)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "image", "image": sample.filepath}  # Pass the PIL Image directly                    
                    # {"type": "image", "image": image}  # Pass the PIL Image directly
                ]
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text],
            images=[image],  # Pass the PIL Image directly
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        output_ids = self.model.generate(**inputs, max_new_tokens=8192, do_sample=False)
        generated_ids = [output_ids[i][len(input_ids):] for i, input_ids in enumerate(inputs.input_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

        # Get image dimensions and convert to float
        input_height = float(sample.metadata.height)
        input_width = float(sample.metadata.width)

        # For VQA, return the raw text output
        if self.operation == "vqa":
            return output_text.strip()

        # For other operations, parse JSON and convert to appropriate format
        parsed_output = self._parse_json(output_text)
        if not parsed_output:
            return None
        
        if self.operation == "classify":
            return self._to_classifications(parsed_output)
        # if self.operation == "detect":
        #     return self._to_detections(parsed_output, input_width, input_height)
        # elif self.operation == "point":
        #     return self._to_keypoints(parsed_output, input_width, input_height)
        # elif self.operation == "segment":
        #     return self._to_polylines(parsed_output, input_width, input_height)
        # elif self.operation == "classify":
        #     return self._to_classifications(parsed_output)

    def predict(self, image, sample=None):
        """Process an image with the model.
        
        A convenience wrapper around _predict that handles numpy array inputs
        by converting them to PIL Images first.
        
        Args:
            image: PIL Image or numpy array to process
            sample: Optional FiftyOne sample containing the image filepath
            
        Returns:
            Model predictions in the appropriate format for the current operation
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self._predict(image, sample)