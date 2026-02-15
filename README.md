# Simulator for PAPI
This repository reproduces the work of PAPI.
PAPI is a Dynamic Parallelism system aimed to analyze the transformer-based generation model (TbGM) inference in a heterogeneous system consisting of xPU and PIMs (Attn-PIM and FC-PIM).

In simulating the PAPI system, the simulator outputs the performance and energy usage of the xPU, while the behavior of PIMs is simulated using a properly modified [Ramulator 2.0](https://github.com/CMU-SAFARI/ramulator2).

This work is adapted from the source code of [AttAcc](https://github.com/scale-snu/attacc_simulator.git). If you are interested in my reproduction approach, please refer to the document "PAPI复现思路.md".

For more details of PAPI, please check the [paper](https://arxiv.org/abs/2502.15470) **PAPI: Exploiting Dynamic Parallelism in Large Language Model Decoding with a Processing-In-Memory-Enabled Computing System** published at [ASPLOS 2025](https://www.asplos-conference.org/asplos2025).

 
## Prerequisites
- Python
- cmake, g++, and clang++ (for building Ramulator2)

PAPI simulator is tested under the following system.

* OS: Ubuntu 22.04.3 LTS (Kernel 6.1.45)
* Compiler: g++ version 12.3.0
* python 3.8.8

We use a similar build system (CMake) as original Ramulator 2.0, which automatically downloads following external libraries.
- [argparse](https://github.com/p-ranav/argparse)
- [spdlog](https://github.com/gabime/spdlog)
- [yaml-cpp](https://github.com/jbeder/yaml-cpp)


## How to install
Clone the Github repository & Build Ramulator2

```bash
$ git clone https://github.com/Lucyyannn/PAPI.git
$ cd PAPI/ramulator2
$ mkdir build
$ cd build
$ cmake ..
$ make -j
$ cp ramulator2 ../ramulator2
$ cd ../../
``` 

## How to run

### Run PAPI simulator 
```bash
$ export PYTHONPATH=$PYTHONPATH:$PWD
$ python main.py --gpu {} --ngpu {} --model {} --word {} --batch {} 
$ python main.py --help

    ## GPU 基础配置
    parser.add_argument("--gpu", type=str, default='A100a', help="GPU type (A100a, H100)")
    parser.add_argument("--ngpu", type=int, default=8, help="number of GPUs")
    parser.add_argument("--gmemcap", type=int, default=60, help="per-GPU memory capacity (GB)")

    ## LLM 负载配置
    parser.add_argument("--model", type=str, default='GPT-175B', help="model list: GPT-175B")
    parser.add_argument("--word", type=int, default=2, help="Precision: 1(INT8), 2(FP16)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--alpha", type=int, default=3, help="threshold of the AI")
```

### Examples
```bash 
$ python3 main.py --gpu A100a --ngpu 8 --model GPT-175B --batch 8 --alpha 3
```

## Details of the Ramulator for AttAcc
### How to Run
1. Generate PIM command traces for the Transformer-based Generative Model.
```bash
$ cd ramulator2
$ cd trace_gen
$ python3 gen_trace_papi_attn.py
$ python3 gen_trace_papi_fc.py
```

This produces `papi_attn.trace` and `papi_fc.trace` which are GPT-175B traces of attention layer or fc layer respectively.

You can change the configuration by setting arguments as below.
```python
  # gen_trace_papi_attn.py
  parser.add_argument("-dh", "--dhead", type=int, default=128, 
                      help="dhead, default= 128")
  parser.add_argument("-nh", "--nhead", type=int, default=1, 
                      help="Number of heads, default=1")
  parser.add_argument("-l", "--seqlen", type=int, default=2048,
                      help="Sequence length L, default= 2048")
  parser.add_argument("-maxl", "--maxlen", type=int, default=4096, 
                      help="maximum L, default= 4096")
  parser.add_argument("-db", "--dbyte", type=int, default=2, 
                      help="data type (B), default= 2")
  parser.add_argument("-o", "--output", type=str, default="papi_attn.trace", 
                      help="output path")

  # python3 gen_trace_papi_fc.py
  parser.add_argument("-hin", "--h_in", type=int, default=12288, help="input hdim")
  parser.add_argument("-hout", "--h_out", type=int, default=12288, help="output hdim")
  parser.add_argument("-b", "--batch_size", type=int, default=8, help="batch_size")
  parser.add_argument("-o", "--output", type=str, default="papi_fc.trace", help="output path")
```

2. Run Ramulator-AttAcc
```bash
$ ./ramulator2 -f papi_attn.yaml
$ ./ramulator2 -f papi_fc.yaml
```

This will print the total number of DRAM/PIM request and total elapsed memory cycles (`memory_system_cycles`).

The command log will be generated in `log` directory.





