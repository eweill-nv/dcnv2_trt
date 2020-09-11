# DCNv2 ONNX to TensorRT

This repo provides the code necessary to build a custom TensorRT plugin for a network containing the DCNv2 layer types.  The setup here assumes that you have already created your ONNX model and just want to convert to TensorRT.

> NOTE: CUDA kernels for this plugin are slightly modified from [tensorRTIntegrate](https://github.com/dlunion/tensorRTIntegrate/blob/master/src/onnxplugin/plugins/DCNv2.cu)
> with a re-write of the TensorRT API to use the plugin itself.

## Overview

Just as a quick overview of what we want to do, once we have our container setup properly with all necessary packages and have our original ONNX model, we can simply follow these few commands to create the TensorRT engine using the custom plugin.

```sh
# Convert attributes of ONNX model
$ python scripts/insert_dcn_plugin.py --input=/models/original.onnx --output=/models/modified.onnx

# Build the TensorRT plugin
$ make -j$(nproc)

# Use trtexec to build and execute the TensorRT engine
$ trtexec --onnx=/models/modified.onnx --plugins=build/dcn_plugin.so --workspace=2000 --saveEngine=/models/dcnv2_trt_fp32.engine
# OR (for FP16)
$ trtexec --onnx=/models/modified.onnx --plugins=build/dcn_plugin.so --workspace=2000 --saveEngine=/models/dcnv2_trt_fp16.engine --fp16
```

Further explanations and customizations are shown below for a more detailed account of what's going on behind the scenes.

## Setup

This material was built on top of the [TensorRT NGC image](https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt) and tested for functionality.  TensorRT container versions 20.07 and 20.08 were used for testing of this plugin.  We will also need to download [OSS TensorRT](https://github.com/nvidia/TensorRT) so that we can use ONNX GraphSurgeon to make some slight modifications to our ONNX model file.

### With Dockerfile

The easiest way to get started is to use the provided [Dockerfile](Dockerfile) to create the Docker image with all the dependencies pre-installed.  To do that, follow these steps:

```sh
# Build the docker image
$ bash scripts/docker/build.sh

# Launch an interactive container (you will need to provide the full path to the directory containing your ONNX model, /models in this example)
$ bash script/docker/launch.sh /models
```

You should now be inside the container with all of the dependencies installed.  You should be able to run the commands above to modify the ONNX model, build the TensorRT plugin, and then create/run the TensorRT engine for the model.

### Without Dockerfile

If you don't choose to use the Dockerfile, it will be a little bit more work upfront.  You will need to clone the TensorRT NGC container manually and install some dependencies before you are able to run the commands for ONNX model conversion, plugin building, and TensorRT engine creation.

The following commands should reproduce the environment that the Dockerfile creates:

```sh
# Clone the TensorRT container
$ docker pull nvcr.io/nvidia/tensorrt:20.08-py3

# Launch the container
$ docker run --gpus all \
    -v <path_to_onnx_model>:/models \
    --name <name_for_docker_container> \
    --network host \
    --rm \
    -i -t \
    nvcr.io/nvidia/tensorrt:20.08-py3 \
    bash
```

Once inside the container, you will need to install a few things to before getting started:

```sh
# Clone OSS TensorRT
$ git clone -b master https://github.com/nvidia/TensorRT TensorRT
$ cd TensorRT/tools/onnx-graphsurgeon
$ make install
$ cd -

# Install Python bindings for TensorRT
$ /opt/tensorrt/python/python_setup.sh
```

This should give you the same environment that the Dockerfile above will give you.  Then you should be able to go through the process of modifying the ONNX model, building the TensorRT plugin, and creating the TensorRT engine for the model.

## ONNX Model Conversion

We will need to do a slight conversion to our ONNX model so that we are able to convert it to a TensorRT engine.  The first modification that we will make (which doesn't theoretically have to be done, but makes everything easier) is to replace the ONNX `Plugin` node with a more meaningful `DCNv2_TRT` node.  At this point, this is just a placeholder since ONNX doesn't know how to interpret the DCNv2 layer anyway.  To do that, we are going to use [ONNX-GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/master/tools/onnx-graphsurgeon).

```python
dcn_nodes = [node for node in graph.nodes if node.op == "Plugin"]
for node in dcn_nodes:
	node.op = "DCNv2_TRT"
```

This will simply rename all of the `Plugin` nodes to `DCNv2_TRT` and make them easier to find with our TensorRT plugin.

The second thing (arguably more important) is to convert the attributes of the layer from a string into the useable dictionary for the TensorRT plugin to use.  Before this conversion, our attributes have 2 fields (`info` and `name`).  The `info` field is a string of the following form:

```
{"dilation": [1, 1], "padding": [1, 1], "stride": [1, 1], "deformable_groups": 1}
```

What we actually want is to separate this string into individual attributes so they don't have to be parsed as a string by the TensorRT plugin creator (much more difficult).  So we modify the ONNX graph with something similar to the following:

```python
# For each of the "Plugin" nodes
attrs = json.loads(node.attrs["info"])
node.attrs.update(attrs)
del node.attrs["info"]
```

The [insert_dcn_plugin.py script](scripts/insert_dcn_plugin.py) provided with this repo does exactly this and only requires the user to provide the path to the input ONNX model and name of the output model.  It can be used as follows:

```sh
python scripts/insert_dcn_plugin.py --input models/<original_onnx_model>.onnx --output models/<modified_onnx_model>.onnx
```

## Plugin

Now that we have all our packages installed, we can now go ahead with building our TensorRT plugin that we will use to convert the ONNX model to TensorRT.  We have provided a Makefile that compiles the `.cpp` and `.cu` files as well as links the appropriate libraries (including `-lcudart`, `-lcublas`, `-lnvinfer`, `-lnvparser`, etc.).  To use it, you can simply run:

```sh
$ make -j$(nproc)
```

This will produce the necessary shared object file (i.e. `build/dcn_plugin.so`) that will be used to build the TensorRT engine.

## TensorRT Engine

To create the TensorRT engine and test it with the plugin, we will use `trtexec`.  This will allow us to run synthetic data through the network to get an idea of the speed of the network as well as output a serialized engine that we can use later.  Note that we are giving a large enough workspace as the plugin itself will use some of the workspace to determine the best operations to perform to create the most optimized engine.

```sh
$ trtexec --onnx=<path_to_onnx_model>.onnx --plugins=build/dcn_plugin.so --workspace=2000 --saveEngine=<path_to_output_trt_engine>.engine
```

## Explanation of TensorRT Plugin Development

Now that we have created the TensorRT engine, let's dive a little deeper into how we were able to do that.


### IPluginV2DynamicExt

The first thing that we want to point out is the we are going to base our Plugin off of the [IPluginV2DynamicExt](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_plugin_v2_dynamic_ext.html) class which will give us the ability to use alot of the functionality that TensorRT already has built in.  You can see where we built our plugin class around the IPluginV2DynamicExt class [here](DCNv2Plugin.h#L60).

The first thing we want to do is to create our constructor and destructor for our TensorRT plugin (in this case, `DCNv2PluginDynamic`).  You can see an example of that [here](DCNv2Plugin.h#L63-69):

```
DCNv2PluginDynamic();

DCNv2PluginDynamic(const void* data, size_t length, const std::string& name);
			
DCNv2PluginDynamic(DCNv2Parameters param, const std::string& name);
			
~DCNv2PluginDynamic() override;
```

Note that here, we have 2 different ways that a `DCNv2PluginDynamic` can be created: either passing in the data as a pointer and reading each value separately, or simply passing it in as a `mParam` structure with all of the data already included in the right format.


#### Methods

We have a few methods that are part of the `IPluginV2DynamicExt` class that we want to override so that we can modify if necessary:

- [clone()](DCNv2Plugin.cpp#L67-73): copies over all internal plugin parameters and returns a new plugin object with these parameters
- [getOutputDimensions()](DCNv2Plugin.cpp#L75-88): computes the dimensions of the output tensor from dimensions of the input tensor
- [supportsFormatCombination()](DCNv2Plugin.cpp#L90-100): determines supported data types
- [configurePlugin()](DCNv2Plugin.cpp#L102-114): configure the layer
- [getWorkspaceSize()](DCNv2Plugin.cpp#L116-129): find the workspace size required by the layer (it is still necessary to provide the `--workspace` flag to `trtexec` as well)
- [enqueue()](DCNv2Plugin.cpp#L131-137): execute the layer

Next, we have a few of the methods that are part of `IPluginv2Ext` that we want to override as well for our functionality:

- [getOutputDataType()](DCNv2Plugin.cpp#L140-144): returns the datatype of the plugin output (in this case, either kFLOAT or kHALF)
- [attachToContext()](DCNv2Plugin.cpp#L148-153): attach the plugin to an execution context and graph plugin access to context resources (use of cuBLAS/cuDNN/etc.)
- [detatchFromContext()](DCNv2Plugin.cpp#L156): detach the plugin from its execution context

Lastly, we have a few of the methods that are part of `IPluginV2` that we want to override for the same reason:

- [getPluginType()](DCNv2Plugin.cpp#L159-162): return the type for the plugin (matches the plugin name returned by the plugin creator)
- [getPluginVersion()](DCNv2Plugin.cpp#L164-167): returns the plugin version (should also match the plugin version returned by the plugin creator)
- [getNbOutputs()](DCNv2Plugin.cpp#L169-172): returns number of outputs for the layer
- [initialize()](DCNv2Plugin.cpp#L174-177): initialize the layer for execution (called when the engine is created)
- [terminate()](DCNv2Plugin.cpp#L179): releases resources aqcuired during plugin layer initialization (called when engine is destroyed)
- [getSerializationSize()](DCNv2Plugin.cpp#L181-188): returns size of serialization buffer necessary
- [serialize()](DCNv2Plugin.cpp#L190-200): serialize the layer
- [destroy()](DCNv2Plugin.cpp#L202-205): destroy the plugin object
- [setPluginNamespace()](DCNv2Plugin.cpp#L207-210): set the namespace for the plugin object
- [getPluginNamespace()](DCNv2Plugin.cpp#L212-215): return the namespace for the plugin object

More information about these plugins can be found [here](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_plugin_v2_dynamic_ext.html) in the TensorRT documentation.

#### Members

As part of our `IPluginV2DynamicExt` instantiated class, we also want to create a few member variables that will help us with our plugin implementation.  These can be found [here](DCNv2Plugin.h#L121-132)

We have the following variables defined here:

- mLayerName: given name for the layer (how it shows up in the graph)
- mNamespace: namespace in which the layer resides
- cublasHandle_: handle to the cuBlas context
- cudnnHandle_: handle to the cuDNN context
- mType: layer type (in this case, either kFLOAT or kHALF)
- input1_shape: shape of first input to DCNv2 layer (from Add layer)
- input2_shape: shape of second input to DCNv2 layer (from Conv layer)
- weights_shape: shape of weights for DCNv2 layer
- output_shape: shape of output for DCNv2 layer
- [mParam](DCNv2Plugin.h#L37-40): structure container attributes for DCNv2 layer
- mDeviceWeights: variable for weights on GPU for DCNv2 layer
- mDeviceBiases: variable for biases on GPU for DCNv2 layer

### IPluginCreator

For user implemented layers, we need to also instantiate another class (i.e. `DCNv2PluginDynamicCreator`) which is going to be part of the `IPluginCreator` class with the following methods:

The first thing we want to do (as before) is create our constructor and destructor for our TensorRT plugin creator (in this case, `DCNv2PluginDynamicCreator`).  An example of that can be found [here](DCNv2Plugin.h#L147) with the declaration [here](DCNv2Plugin.cpp#L218-231).

#### Methods

- getTensorRTVersion(): return version of API the plugin creator was compiled with
- [getPluginName()](DCNv2Plugin.cpp#L233-236): return plugin name
- [getPluginVersion()](DCNv2Plugin.cpp#L238-241): return plugin version
- [getFieldNames()](DCNv2Plugin.cpp#L243-246):  return list of fields to be passed to createPlugin
- [createPlugin()](DCNv2Plugin.cpp#L248-311): return plugin object
- [deserializePlugin()](DCNv2Plugin.cpp#L313-319): called during deserialization of plugin layer
- [setPluginNamespace()](DCNv2Plugin.cpp#L321-324): set namespace for plugin creator based on plugin library
- [getPluginNamespace()](DCNv2Plugin.cpp#L326-329): return namespace of plugin creator object

#### Members

- mFC: contains information about the PluginFieldCollection
- mPluginAttributes: contains information about attributes of the plugin
- mNamespace: namespace in which the layer reside

More information about the plugin creator can be found [here](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_plugin_creator.html) in the TensorRT documentation.

### DCNv2 Plugin Specifics

Now that we've discussed the specifics of what needs to be implemented, let's talk about what needed to be designed specifically for the DCNv2 TensorRT plugin (we point out the important ones here).

#### DCNv2PluginDynamic Methods

##### DCNv2PluginDynamic Constructor

```cpp
DCNv2PluginDynamic::DCNv2PluginDynamic(const void* data, size_t length, 
	const std::string& name)
	: mLayerName(name)
{
	const char* d = reinterpret_cast<const char*>(data), *a = d;

	for (int i = 0 ; i < DILATION_DIM ; i++)
		mParam.dilation.push_back(read<int>(d));
	for (int i = 0 ; i < PADDING_DIM ; i++)
		mParam.padding.push_back(read<int>(d));
	for (int i = 0 ; i < STRIDE_DIM ; i++)
		mParam.stride.push_back(read<int>(d));
	mParam.deformable_groups = read<int>(d);
}

DCNv2PluginDynamic::DCNv2PluginDynamic(DCNv2Parameters param, const std::string& name)
	: mParam{param}, mLayerName(name)
{}
```

Notice that we have 2 constructors here that work on different types of data.  The first one takes in a pointer to the data itself as a string and parses out each of the attributes manually.  The second takes in a `DCNv2Parameters` structure that already has all the attributes of the layer set properly.

##### DCNv2PluginDynamic configurePlugin

```cpp
void DCNv2PluginDynamic::configurePlugin(const DynamicPluginTensorDesc* inputs, int nbInputs,
	const DynamicPluginTensorDesc* outputs, int nbOutputs)
{
	// Validate input arguments
	assert(nbInputs == 4);
	assert(nbOutputs == 1);

	mType = inputs[0].desc.type;
	input1_shape = inputs[0].desc.dims;
	input2_shape = inputs[1].desc.dims;
	weights_shape = inputs[2].desc.dims;
	output_shape = outputs[0].desc.dims;
}
```

In this method, we set the type of the layer (kFLOAT or kHALF) as well as the input shapes to the layer (`input1_shape` and `input2_shape`).  We also set the shape of the weights here as well as the shape of the output for the layer.

##### DCNv2PluginDynamic Enqueue

```cpp
int DCNv2PluginDynamic::enqueue(const PluginTensorDesc* inputDesc,
	const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs,
	void* workspace, cudaStream_t stream)
{
	enqueue_call(inputs, outputs, workspace, stream, mType, input1_shape, input2_shape,
		weights_shape, output_shape, mType, cublasHandle_, mParam);
}
```

In the `enqueue` method (which is the method that actually executes the layer), we call the `enqueue_call` function which will in turn call the appropriate CUDA kernels.  Here we pass the appropriate shapes as well as inputs and outputs to the layer.

##### DCNv2PluginDynamic getSerializationSize

```cpp
size_t DCNv2PluginDynamic::getSerializationSize() const
{
	return DILATION_DIM * sizeof(int)	// dilation
		+ PADDING_DIM * sizeof(int)	// padding
		+ STRIDE_DIM * sizeof(int)	// stride
		+ 1				// deformable
		;
}
```

With this method, we need to let TensorRT the extra parameters if will need to serialize along with the rest of the model and the size of that.  Here we add the size of the `dilation`, `padding`, `stride`, and `deformable_groups` to the serialization size.

##### DCNv2PluginDynamic Serialize

```cpp
void DCNv2PluginDynamic::serialize(void* buffer) const
{
	char* d = reinterpret_cast<char*>(buffer), *a = d;
	for (int i = 0 ; i < DILATION_DIM ; i++)
		write(d, mParam.dilation[i]);
	for (int i = 0 ; i < PADDING_DIM ; i++)
		write(d, mParam.padding[i]);
	for (int i = 0 ; i < STRIDE_DIM ; i++)
		write(d, mParam.stride[i]);
	write(d, mParam.deformable_groups);
}
```

Here we take those important attributes and write them out to our engine file during the serialization process.

#### DCNv2PluginDynamicCreator Methods

##### DCNv2PluginDynamicCreator Constructor

```cpp
DCNv2PluginDynamicCreator::DCNv2PluginDynamicCreator()
{
	mPluginAttributes.emplace_back(PluginField("dilation", nullptr,
		PluginFieldType::kINT32, 2));
	mPluginAttributes.emplace_back(PluginField("padding", nullptr,
		PluginFieldType::kINT32, 2));
	mPluginAttributes.emplace_back(PluginField("stride", nullptr,
		PluginFieldType::kINT32, 2));
	mPluginAttributes.emplace_back(PluginField("deformable_groups", nullptr,
		PluginFieldType::kINT32, 1));

	mFC.nbFields = mPluginAttributes.size();
	mFC.fields = mPluginAttributes.data();
}
```

Notice with this constructor, we are setting the PluginField for each of the attributes of the DCNv2 layer type.  We are also setting the number of fields and field data that we will use later when we configure the plugin.

##### DCNv2PluginDynamicCreator CreatePlugin

```cpp
IPluginV2* DCNv2PluginDynamicCreator::createPlugin(const char* name,
	const PluginFieldCollection* fc)
{
	std::vector<int> dilation;
	std::vector<int> padding;
	std::vector<int> stride;
	int deformable_groups;
	const PluginField* fields = fc->fields;

	for (int i = 0 ; i < fc->nbFields ; i++)
	{
		const char* attrName = fields[i].name;
		if (!strcmp(attrName, "dilation"))
		{
			ASSERT(fields[i].type == PluginFieldType::kINT32);
			int size = fields[i].length;
			dilation.reserve(size);
			const auto* d = static_cast<const int*>(fields[i].data);
			for (int j = 0 ; j < size ; j++)
			{
				dilation.push_back(*d);
				d++;
			}
		}
		else if (!strcmp(attrName, "padding"))
		{
			ASSERT(fields[i].type == PluginFieldType::kINT32);
			int size = fields[i].length;
			padding.reserve(size);
			const auto* p = static_cast<const int*>(fields[i].data);
			for (int j = 0 ; j < size ; j++)
			{
				padding.push_back(*p);
				p++;
			}
		}
		else if (!strcmp(attrName, "stride"))
		{
			ASSERT(fields[i].type == PluginFieldType::kINT32);
			int size = fields[i].length;
			stride.reserve(size);
			const auto* s = static_cast<const int*>(fields[i].data);
			for (int j = 0 ; j < size ; j++)
			{
				stride.push_back(*s);
				s++;
			}
		}
		else if (!strcmp(attrName, "deformable_groups"))
		{
			ASSERT(fields[i].type == PluginFieldType::kINT32);
			deformable_groups = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
		}
	}

	DCNv2Parameters dcnv2Params;
	dcnv2Params.dilation = dilation;
	dcnv2Params.padding = padding;
	dcnv2Params.stride = stride;
	dcnv2Params.deformable_groups = deformable_groups;

	DCNv2PluginDynamic* p = new DCNv2PluginDynamic(dcnv2Params, name);
	return p;
}
```

During the `createPlugin` method, we take all of the attributes from the model (i.e. `dilation`, `padding`, `stride`, and `deformable_groups`) and put them into a parameter structure that is used to create the plugin itself.


## Licenses and Agreements
* [Apache 2.0 License](LICENSE)
* [Individual Contributor License Agreement (CLA)](https://gist.github.com/alex3165/0d70734579a542ad34495d346b2df6a5)
