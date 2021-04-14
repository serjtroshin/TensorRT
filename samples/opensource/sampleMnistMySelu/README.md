# Implementing Selu in TensorRT with a custom plugin

## Description

This sample, sampleMnistmMySelu, converts a model trained on the `MNIST dataset` in Open Neural Network Exchange (ONNX) format to a TensorRT network and runs inference on the network.
This model was trained in PyTorch and it contains custom Selu layers instead of Relu layers.<br/>
Model with Selu layers training script in PyTorch: [link](https://github.com/serjtroshin/mnist-selu-pytorch)<br/>
Original model with usual Relu layers: [link](https://github.com/pytorch/examples/tree/master/mnist)

This plugin can be found at `TensorRT/plugin/mySeluPlugin`.

ONNX is a standard for representing deep learning models that enables models to be transferred between frameworks.

## How does this sample work?

This sample implements Selu layer and runs a TensorRT engine on an ONNX model of MNIST trained with Selu layer.

Specifically, this sample:
- [Converts the ONNX model with custom layer to a TensorRT network](#converting-the-onnx-model-to-a-tensorrt-network)
- [Builds an engine with custom layer](#building-an-engine)
- [Runs inference using the generated TensorRT network](#running-inference)


To run inference using the created engine, see [Performing Inference In C++](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#perform_inference_c).

**Note:** It’s important to preprocess the data and convert it to the format accepted by the network. In this example, the sample input is in PGM (portable graymap) format. The model expects an input of image `1x28x28` scaled to between `[0,1]`. 

**Note2:** Additional preprocessing needs to be applied to the data before putting it to the NN input due to the same normalization preprocessing were used when model was trained [transforms.Normalize((0.1307,), (0.3081,))](https://github.com/pytorch/examples/tree/master/mnist):

```
const float PYTORCH_NORMALIZE_MEAN = 0.1307;
const float PYTORCH_NORMALIZE_STD = 0.3081;
hostDataBuffer[i] = ((1.0 - float(fileData[i] / 255.0)) - PYTORCH_NORMALIZE_MEAN) / PYTORCH_NORMALIZE_STD;
```


## Running the sample

1. Run `mkdir build && cd build && cmake ..` in `<TensorRT root directory>`, then `cd samples/opensource/sampleMnistMySelu && make`. 
The binary named `sample_mnist_my_selu` will be in `<TensorRT root directory>/build/out` directory.
To check it on random digit from the MNIST dataset:
	```
	LD_LIBRARY_PATH=. ./sample_mnist_my_selu -d $TRT_DATADIR
	```
where inside `$TRT_DATADIR` should be located `mnist_my_selu.onnx` and `[0-9].pgm` files.

3.  Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:
	```
	[04/14/2021-13:07:42] [I] Input:
	[04/14/2021-13:07:42] [I] @@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@#**    -@@@@@@@@
	@@@@@@@@@@*:.       :-@@@@@@
	@@@@@@@@+.            :@@@@@
	@@@@@@@+               :%@@@
	@@@@@@@.                *@@@
	@@@@@%.     .@@@@*:    =@@@@
	@@@@@*      .#@@@@*   -@@@@@
	@@@@@*        -@@=  .=#@@@@@
	@@@@@@:        ..  :@@@@@@@@
	@@@@@@#-.:        :@@@@@@@@@
	@@@@@@@@@@-       .%@@@@@@@@
	@@@@@@@@@@%:       .@@@@@@@@
	@@@@@@@@@*   .=     *@@@@@@@
	@@@@@@@@@:  .%@+    :@@@@@@@
	@@@@@@@@#   %@@%.    -@@@@@@
	@@@@@@@#   #@@@@%    .@@@@@@
	@@@@@@@+   .+%%%#    :@@@@@@
	@@@@@@@@.           .%@@@@@@
	@@@@@@@@#.        ::@@@@@@@@
	@@@@@@@@@#     - *@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	@@@@@@@@@@@@@@@@@@@@@@@@@@@@

	[04/14/2021-13:07:42] [I] Output:
	[04/14/2021-13:07:42] [I]  Prob 0  0.0000 Class 0:
	[04/14/2021-13:07:42] [I]  Prob 1  0.0000 Class 1:
	[04/14/2021-13:07:42] [I]  Prob 2  0.0000 Class 2:
	[04/14/2021-13:07:42] [I]  Prob 3  0.0000 Class 3:
	[04/14/2021-13:07:42] [I]  Prob 4  0.0000 Class 4:
	[04/14/2021-13:07:42] [I]  Prob 5  0.0000 Class 5:
	[04/14/2021-13:07:42] [I]  Prob 6  0.0000 Class 6:
	[04/14/2021-13:07:42] [I]  Prob 7  0.0000 Class 7:
	[04/14/2021-13:07:42] [I]  Prob 8  1.0000 Class 8: **********
	[04/14/2021-13:07:42] [I]  Prob 9  0.0000 Class 9:
	[04/14/2021-13:07:42] [I]
	&&&& PASSED TensorRT.sample_mnist_my_selu # ./sample_mnist_my_selu -d /workspace/TensorRT/tensorrt/data/mnist
	```

	This output shows that the sample ran successfully; PASSED.


# Additional resources

The following resources provide a deeper understanding about the ONNX project and MNIST model:

**ONNX**
- [GitHub: ONNX](https://github.com/onnx/onnx)
- [Github: ONNX-TensorRT Open source parser](https://github.com/onnx/onnx-tensorrt)

**Models**
- [MNIST - Handwritten Digit Recognition](https://github.com/onnx/models/tree/master/mnist)
- [GitHub: ONNX Models](https://github.com/onnx/models)

**Documentation**
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The C++ API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#c_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.


# Changelog

April 2020
This `README.md` file was recreated, updated and reviewed.


# Known issues

There are no known issues in this sample.
