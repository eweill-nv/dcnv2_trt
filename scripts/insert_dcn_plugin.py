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


import onnx_graphsurgeon as gs
import argparse
import onnx
import json

def process_graph(graph):
    dcn_nodes = [node for node in graph.nodes if node.op == "Plugin"]
    for node in dcn_nodes:
        node.op = "DCNv2_TRT"
        attrs = json.loads(node.attrs["info"])
        node.attrs.update(attrs)
        del node.attrs["info"]
    return graph

def main():
    parser = argparse.ArgumentParser(description="Insert DCNv2 plugin node into ONNX model")
    parser.add_argument("-i", "--input",
            help="Path to ONNX model with 'Plugin' node to replace with DCNv2_TRT",
            default="models/centertrack_DCNv2_named.onnx")
    parser.add_argument("-o", "--output",
            help="Path to output ONNX model with 'DCNv2_TRT' node",
            default="models/modified.onnx")

    args, _ = parser.parse_known_args()
    graph = gs.import_onnx(onnx.load(args.input))
    graph = process_graph(graph)
    onnx.save(gs.export_onnx(graph), args.output)

if __name__ == '__main__':
    main()
