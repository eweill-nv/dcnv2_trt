#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorrt:20.08-py3
FROM ${FROM_IMAGE_NAME}

#
# Install OSS TensorRT (specifically for ONNX-GraphSurgeon)
#
WORKDIR /workspace
RUN git clone -b master https://github.com/nvidia/TensorRT TensorRT \
    && cd TensorRT/tools/onnx-graphsurgeon \
    && make install \
    && cd -

#
# Install Python bindings for TensorRT
#
RUN /opt/tensorrt/python/python_setup.sh

#
# Copy DCNv2 code into the container
#
WORKDIR /workspace/dcnv2_trt
COPY . .
