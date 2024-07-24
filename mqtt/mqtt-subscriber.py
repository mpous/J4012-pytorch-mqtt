#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import print_function

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
# MQTT needs
import paho.mqtt.client as mqtt
from PIL import Image
import cv2
import base64
import json
from io import BytesIO

# Use autoprimaryctx if available (pycuda >= 2021.1) to
# prevent issues with other modules that rely on the primary
# device context.
try:
    import pycuda.autoprimaryctx
except ModuleNotFoundError:
    import pycuda.autoinit

from PIL import ImageDraw

from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES

import sys, os

sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common
from downloader import getFilePath

TRT_LOGGER = trt.Logger()

# MQTT settings
MQTT_BROKER = os.environ['mqtt_broker']     # 192.168.1.156
print(MQTT_BROKER)
MQTT_PORT = os.environ['mqtt_port']         # 1883
print(MQTT_PORT)
MQTT_SUB_TOPIC = os.environ['mqtt_topic']   # "balena/site/area/line/cell/camera/raw"
MQTT_PUB_TOPIC = "balena/site/area/line/cell/camera/model_name/model_version/inference"
MQTT_PUB_TOPIC_ML = "balena/site/area/line/cell/camera/TensorRT/v8502/inference"
MQTT_PUB_TOPIC_IMG = "balena/site/area/line/cell/camera/inference"


def draw_bboxes(image_raw, bboxes, confidences, categories, all_categories, bbox_color="blue"):
    """Draw the bounding boxes on the original input image and return it.

    Keyword arguments:
    image_raw -- a raw PIL Image
    bboxes -- NumPy array containing the bounding box coordinates of N objects, with shape (N,4).
    categories -- NumPy array containing the corresponding category for each object,
    with shape (N,)
    confidences -- NumPy array containing the corresponding confidence for each object,
    with shape (N,)
    all_categories -- a list of all categories in the correct ordered (required for looking up
    the category name)
    bbox_color -- an optional string specifying the color of the bounding boxes (default: 'blue')
    """
    
    if image_raw is None:
        raise ValueError("image_raw cannot be None")
    if bboxes is None or confidences is None or all_categories is None:
        raise ValueError("boxes, scores, and classes cannot be None")
    if categories is None:
        categories = {}  # Handle None categories with an empty dictionary
    

    draw = ImageDraw.Draw(image_raw)
    print(bboxes, confidences, categories)
    for box, score, category in zip(bboxes, confidences, categories):
        x_coord, y_coord, width, height = box
        left = max(0, np.floor(x_coord + 0.5).astype(int))
        top = max(0, np.floor(y_coord + 0.5).astype(int))
        right = min(image_raw.width, np.floor(x_coord + width + 0.5).astype(int))
        bottom = min(image_raw.height, np.floor(y_coord + height + 0.5).astype(int))

        draw.rectangle(((left, top), (right, bottom)), outline=bbox_color)
        draw.text((left, top - 12), "{0} {1:.2f}".format(all_categories[category], score), fill=bbox_color)

    return image_raw


# Function to convert numpy data to native Python types for JSON serialization
def numpy_to_native(data):
    if isinstance(data, np.generic):
        return data.item()  # Use item() for converting numpy scalar types
    elif isinstance(data, np.ndarray):
        return data.tolist()  # Use tolist() for converting numpy arrays to Python list
    else:
        return data



def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            common.EXPLICIT_BATCH
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser, trt.Runtime(
            TRT_LOGGER
        ) as runtime:
            config.max_workspace_size = 1 << 28  # 256MiB
            builder.max_batch_size = 1
            # Parse model file

        # subscribe to MQTT to get the image
    

            if not os.path.exists(onnx_file_path):
                print(
                    "ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.".format(onnx_file_path)
                )
                exit(0)
            print("Loading ONNX file from path {}...".format(onnx_file_path))
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")




                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [1, 3, 608, 608]
            print("Completed parsing of ONNX file")
            print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def image_to_base64(image_path):
    # Open the image file
    with open(image_path, "rb") as image_file:
        # Read the image file in binary format
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode('utf-8')


# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    # Subscribe to the topic
    client.subscribe(MQTT_SUB_TOPIC)


def on_message(client, userdata, msg):
    print(f"Received message from topic: {msg.topic} ")
    
    # Decode the base64 string to an image
    img_data = base64.b64decode(msg.payload)
    img = Image.open(BytesIO(img_data))
    img_jpg = img.convert('RGB')
    img_jpg.save("/usr/src/tensorrt/samples/python/yolov3_onnx/mqtt-image.jpg", "JPEG")

    inferenceImage()


def inferenceImage():
    # Try to load a previously generated YOLOv3-608 network graph in ONNX format:
    onnx_file_path = "yolov3.onnx"
    engine_file_path = "yolov3.trt"


    # Download a dog image and save it to the following file path:
    input_image_path = getFilePath("/usr/src/tensorrt/samples/python/yolov3_onnx/mqtt-image.jpg")

    #input_image_path = getFilePath("mqtt-image.jpg")
    # Two-dimensional tuple with the target network's (spatial) input resolution in HW ordered
    input_resolution_yolov3_HW = (608, 608)
    # Create a pre-processor object by specifying the required input resolution for YOLOv3
    preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)
    # Load an image from the specified input path, and return it together with  a pre-processed version
    image_raw, image = preprocessor.process(input_image_path)
    # Store the shape of the original input image in WH format, we will need it for later
    shape_orig_WH = image_raw.size

    # Output shapes expected by the post-processor
    output_shapes = [(1, 255, 19, 19), (1, 255, 38, 38), (1, 255, 76, 76)]
    # Do inference with TensorRT
    trt_outputs = []
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Do inference
        print("Running inference on image {}...".format(input_image_path))
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        inputs[0].host = image
        trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
    trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]

    postprocessor_args = {
        "yolo_masks": [(6, 7, 8), (3, 4, 5), (0, 1, 2)],  # A list of 3 three-dimensional tuples for the YOLO masks
        "yolo_anchors": [
            (10, 13),
            (16, 30),
            (33, 23),
            (30, 61),
            (62, 45),  # A list of 9 two-dimensional tuples for the YOLO anchors
            (59, 119),
            (116, 90),
            (156, 198),
            (373, 326),
        ],
        "obj_threshold": 0.6,  # Threshold for object coverage, float value between 0 and 1
        "nms_threshold": 0.5,  # Threshold for non-max suppression algorithm, float value between 0 and 1
        "yolo_input_resolution": input_resolution_yolov3_HW,
    }

    postprocessor = PostprocessYOLO(**postprocessor_args)

    # Run the post-processing algorithms on the TensorRT outputs and get the bounding box details of detected objects
    boxes, classes, scores = postprocessor.process(trt_outputs, (shape_orig_WH))
    # Draw the bounding boxes onto the original input image and save it as a PNG file
    try:
        obj_detected_img = draw_bboxes(image_raw, boxes, scores, classes, ALL_CATEGORIES)
    except ValueError as e:
        print(e)
        obj_detected_img = image_raw

    output_image_path = "mqtt_bboxes.png"
    obj_detected_img.save(output_image_path, "PNG")

    print("Saved image with bounding boxes of detected objects to {}.".format(output_image_path))

    # Encode the processed image back to a base64 string
    base64_string = image_to_base64(output_image_path)
    print("Publishing MQTT messages after inferences...")

    base64_image_message = json.dumps({"base64Image": base64_string})

    # Generate detections JSON structure
    detections = [
            {
                "class": ALL_CATEGORIES[numpy_to_native(cls)],
                "score": numpy_to_native(score),
                "boundingBox": {
                    "x_min": numpy_to_native(box[0]),
                    "y_min": numpy_to_native(box[1]),
                    "x_max": numpy_to_native(box[2]),
                    "y_max": numpy_to_native(box[3]),
                }
            } for cls, score, box in zip(classes, scores, boxes)
    ]

    # Serialize the Python object to a JSON formatted string
    json_output = json.dumps({"detections": detections}, indent=4)

    print(json_output)

    client.publish(MQTT_PUB_TOPIC_ML, json_output)
    client.publish(MQTT_PUB_TOPIC_IMG, base64_image_message)


# Create a TensorRT engine for ONNX-based YOLOv3-608 and run MQTT subcription
# MQTT subscription

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

print("Ready to connect...")
client.connect(MQTT_BROKER, int(MQTT_PORT), 60)
print("Connecting...")

# Blocking call that processes network traffic, dispatches callbacks, and handles reconnecting.
client.loop_forever()
