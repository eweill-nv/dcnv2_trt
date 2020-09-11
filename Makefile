
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

BUILD_DIR := build

TRT_OSS_ROOT ?= /workspace/TensorRT
TRT_INC_DIR ?= /usr/include/x86_64-linux-gnu/
TRT_LIB_DIR ?= /usr/lib/x86_64-linux-gnu/

CUDNN_INC_DIR ?= /usr/include/x86_64-linux-gnu/

CFLAGS = -c -Xcompiler -fPIC -w
CFLAGS += -I$(CURDIR) -I$(TRT_INC_DIR) -I$(TRT_OSS_ROOT)/plugin/common/kernels/ -I$(TRT_OSS_ROOT)/plugin/common/ -I$(TRT_OSS_ROOT)/plugin/ -I$(CUDNN_INC_DIR)

LFLAGS = -shared -lcudart -lcublas -lnvinfer -lnvparsers -L$(TRT_LIB_DIR)
GENCODES = -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75

OBJS := $(patsubst %.cpp, $(BUILD_DIR)/%.o, $(wildcard *.cpp))
CUOBJS := $(patsubst %.cu, $(BUILD_DIR)/%.cu.o, $(wildcard *.cu))

.PHONY: all
all: $(BUILD_DIR)/dcn_plugin.so

$(BUILD_DIR)/%.o: %.cpp
	nvcc $^ $(CFLAGS) -o $@

$(BUILD_DIR)/%.cu.o: %.cu
	nvcc $^ $(GENCODES) $(CFLAGS) -o $@

$(BUILD_DIR)/dcn_plugin.so: $(OBJS) $(CUOBJS)
	nvcc $^ $(LFLAGS) -o $@

# Make sure that the build directory exists before compilation
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(OBJS): | $(BUILD_DIR)
$(CUOBJS): | $(BUILD_DIR)
