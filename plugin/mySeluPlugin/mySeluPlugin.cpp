/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "mySeluPlugin.h"
#include <cstring>
#include <iostream>
#include <vector>

using namespace nvinfer1;

namespace
{
const char* MY_SELU_PLUGIN_VERSION{"1"};
const char* MY_SELU_PLUGIN_NAME{"mySelu"};
} // namespace

PluginFieldCollection mySeluPluginCreator::mFC{};
std::vector<PluginField> mySeluPluginCreator::mPluginAttributes;

mySeluPlugin::mySeluPlugin(): mBatchDim(1) {}

mySeluPlugin::mySeluPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    mBatchDim = read<int>(d);
    ASSERT(d == a + length);
}

int mySeluPlugin::getNbOutputs() const
{
    return 1;
}

int mySeluPlugin::initialize()
{
    return STATUS_SUCCESS;
}

void mySeluPlugin::terminate() {}

Dims mySeluPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    ASSERT(nbInputDims == 1);
    ASSERT(index == 0);
    return inputs[0];
}

size_t mySeluPlugin::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

size_t mySeluPlugin::getSerializationSize() const
{
    // mBatchDim
    return sizeof(int);
}

void mySeluPlugin::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mBatchDim);
    ASSERT(d == a + getSerializationSize());
}

void mySeluPlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, nvinfer1::PluginFormat format, int maxBatchSize)
{
    gLogVerbose << "mySeluPlugin configurePlugin\n";
    ASSERT(nbInputs == 1);
    ASSERT(nbOutputs == 1);

    mBatchDim = 1;
    for (size_t i = 0; i < inputDims->nbDims; ++i) {
        mBatchDim *= inputDims->d[i];
    }
    ASSERT(inputTypes[0] == DataType::kFLOAT);
}

bool mySeluPlugin::supportsFormat(DataType type, PluginFormat format) const
{
    return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
}

const char* mySeluPlugin::getPluginType() const
{
    return MY_SELU_PLUGIN_NAME;
}

const char* mySeluPlugin::getPluginVersion() const
{
    return MY_SELU_PLUGIN_VERSION;
}

void mySeluPlugin::destroy()
{
    delete this;
}

IPluginV2Ext* mySeluPlugin::clone() const
{
    IPluginV2Ext* plugin = new mySeluPlugin(*this);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void mySeluPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* mySeluPlugin::getPluginNamespace() const
{
    return mPluginNamespace;
}

nvinfer1::DataType mySeluPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    return inputTypes[0];
}

bool mySeluPlugin::isOutputBroadcastAcrossBatch(
    int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

bool mySeluPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

// Plugin creator
mySeluPluginCreator::mySeluPluginCreator() {}

const char* mySeluPluginCreator::getPluginName() const
{
    return MY_SELU_PLUGIN_NAME;
}

const char* mySeluPluginCreator::getPluginVersion() const
{
    return MY_SELU_PLUGIN_VERSION;
}

const PluginFieldCollection* mySeluPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2Ext* mySeluPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    mySeluPlugin* plugin = new mySeluPlugin();
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2Ext* mySeluPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    mySeluPlugin* plugin = new mySeluPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
