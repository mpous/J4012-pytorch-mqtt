# Seeed J4012 PyTorch/TensorRT Example

This is a sample for running visual inferencing on the Seeed J4012 reComputer (NVIDIA Jetson Orin NX) hardware using balenaOS.

## Usage

This sample is still under development...

To test the TensorRT is running correctly on the NVIDIA hardware:

- Go to the `/usr/src/tensorrt/samples` folder
- Run `make TARGET=aarch64` - this could take 15 minutes to compile all the samples
- Go to the `/usr/src/tensorrt/bin` folder
- Run `./sample_onnx_mnist`

You should see `Building and running a GPU inference engine for Onnx MNIST` and a bunch of test output, ending with:

`&&&& PASSED TensorRT.sample_onnx_mnist [TensorRT v8502] # ./sample_onnx_mnist`

## More information


This example repo is using the project https://github.com/dusty-nv/jetson-inference, which is based on the [NVIDIA l4T PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch) base image.

However, the "jetson-interface" project fails to build in the container, so you should use the [NVIDIA-supplied examples](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-713/sample-support-guide/index.html#samples) which are located in the container at `/usr/src/tensorrt/samples` as mentioned above. Below is another NVIDIA example that does run in the container.

## Object Detection With The ONNX TensorRT Backend In Python 

Here is a quick start to run this example: (full documentation [here](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-713/sample-support-guide/index.html#yolov3_onnx))

In the container, do the following: (note: `/inference-store` is simply a persistent volume for storing models, etc...)

```
cd /usr/src/tensorrt/samples/python/
python3 downloader.py -d /inference-store -f /usr/src/tensorrt/samples/python/yolov3_onnx/download.yml
cd yolov3_onnx
python3 yolov3_to_onnx.py -d /inference-store
python3 onnx_to_tensorrt.py  -d /inference-store
```

If there is any issue with the `yolov3.weights` file download it manually using `wget https://pjreddie.com/media/files/yolov3.weights` and move the file to the `/inference-store` folder.

If successful, you should execute `python3 onnx_to_tensorrt.py  -d /inference-store` and see:

```
Running inference on image /jetson-inference/python/training/classification/models/samples/python/yolov3_onnx/dog.jpg...
[[134.94005922 219.30816557 184.32604687 324.51474599]
 [ 98.63753808 136.02425953 499.65646314 298.39950069]
 [477.79566252  81.31128895 210.98671105  86.85283442]] [0.9985233  0.99885205 0.93972876] [16  1  7]
Saved image with bounding boxes of detected objects to dog_bboxes.png.
```

Now you can modify `onnx_to_tensorrt.py` to run your own inferences!

## Using MQTT to get images for Object Detection With The ONNX TensorRT Backend

Create the file `mqtt-subscriber.py` with nano and copy the `mqtt/mqtt-subscriber.py` source code from this repository. Paste it there in the nano editor and save the code.

Then, add the Device Variables `mqtt-broker`, `mqtt-port` and `mqtt-topic` on balenaCloud.

![MQTT variables on balenaCloud](https://github.com/mpous/J4012-pytorch-mqtt/assets/173156/e9ef1fc6-5109-4d6a-9ffb-a8e07d8d6f84)

Finally, run:

```
python3 mqtt-subscriber.py -d /inference-store
``` 

And if everything works properly you might see something similar:

```
192.168.1.156
1883
mqtt-subscriber.py:282: DeprecationWarning: Callback API version 1 is deprecated, update to latest version
  client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
Connected with result code 0
Received message from topic: balena/site/area/line/cell/camera/raw 
MQTT image JPG created
Reading engine from file yolov3.trt
[05/09/2024-12:44:26] [TRT] [W] The getMaxBatchSize() function should not be used with an engine built from a network created with NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag. This function will always return 1.
[05/09/2024-12:44:26] [TRT] [W] The getMaxBatchSize() function should not be used with an engine built from a network created with NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag. This function will always return 1.
[05/09/2024-12:44:26] [TRT] [W] The getMaxBatchSize() function should not be used with an engine built from a network created with NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag. This function will always return 1.
[05/09/2024-12:44:26] [TRT] [W] The getMaxBatchSize() function should not be used with an engine built from a network created with NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag. This function will always return 1.
Running inference on image /usr/src/tensorrt/samples/python/yolov3_onnx/mqtt-image.jpg...
[[ 19.11184406  44.34086917 468.70917977 341.92075705]
 [ -0.64684702  49.89250563 459.59584178 431.62784936]] [0.61729218 0.99282281] [56  0]
Saved image with bounding boxes of detected objects to mqtt_bboxes.png.
Publishing MQTT messages after inferences...
{
    "detections": [
        {
            "class": "chair",
            "score": 0.6172921839895628,
            "boundingBox": {
                "x_min": 19.111844060595615,
                "y_min": 44.34086916586583,
                "x_max": 468.7091797705793,
                "y_max": 341.9207570511907
            }
        },
        {
            "class": "person",
            "score": 0.9928228082087157,
            "boundingBox": {
                "x_min": -0.6468470192650599,
                "y_min": 49.89250563283248,
                "x_max": 459.5958417775767,
                "y_max": 431.62784935553987
            }
        }
    ]
}
Received message from topic: balena/site/area/line/cell/camera/raw 
MQTT image JPG created
...

```

You can see here the images of this example.

![Raw image and image with bounding boxes](https://github.com/mpous/J4012-pytorch-mqtt/assets/173156/0164ba4a-7016-40ee-8260-1e60f6e26fae)



## Attribution

* This is in part working thanks of the work of Kudzai Manditereza from HiveMQ, and Alan Boris and Marc Pous from balena.io.
* This is working thanks to the amazing YoloV3 and [Dustin Franklin](https://github.com/dusty-nv) examples.


## Disclaimer

This project is for educational purposes only. Do not deploy it into production or your premises without understanding what you are doing.

