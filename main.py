import argparse
import csv
import os
from src.system import *
from src.type import *
from src.config import *
from src.ramulator_wrapper import *

# PAPI 仿真开关
RAMULATOR = True 

def write_csv(logfile, perfs):
    if logfile is not None:
        firstrow = False
        if not os.path.exists(logfile):
            firstrow = True

        f = open(logfile, 'a')
        wrt = csv.writer(f)
        if firstrow:
            col_name = [
                'model', 'dtype', 'xpu', 'cap', 'bw', 'sys_opb', 'hw', 'cores',
                'pipe_level', 'is parallel', 'power constraint', 'gqa_size',
                'Lin', 'Lout', 'bs', 'required_cap', 's_flops',
                'g_flops', 's_time', 's_matmul', 's_fc', 's_comm', 's_softmax',
                's_act', 's_lnorm', 'g_time (ms)', 'g_matmul', 'g_fc', 'g_comm',
                'g_etc', 'g_qkv_time', 'g_prj_time', 'g_ff_time', 'g2g_comm',
                'c2g_comm', 'g_softmax', 'g_act', 'g_lnorm', 'g_energy (nJ)',
                'g_dram_energy', 'g_l2_energy', 'g_l1_energy', 'g_reg_energy',
                'g_alu_energy', 'g_fc_mem_energy', 'g_fc_comp_energy',
                'g_attn_mem_energy', 'g_attn_comp_energy', 'g_etc_mem_energy',
                'g_etc_comp_energy', 'g_comm_energy'
            ]
            wrt.writerow(col_name)

        for perf in perfs:
            tag, config, time, energy = perf
            info = tag + config + time + energy
            wrt.writerow(info)
        f.close()

def run(system: System,
        batch,
        lin,
        lout,
        power_constraint=False,
        pipe=0,
        parallel=False,
        output_file=None):
    print("--- PAPI Heterogeneous Simulation | Batch {} Lin {} Lout {} ---".
          format(batch, lin, lout))
    
    assert system.model_set, "Need to SetModel"
    perfs = []
    
    system.simulate(batch,
                    lin,
                    lout,
                    perfs=perfs,
                    pipe=pipe,
                    parallel_ff=parallel,
                    power_constraint=power_constraint)
    
    if output_file is not None:
        write_csv(output_file, perfs)

def main():
    parser = argparse.ArgumentParser(
        description="PAPI configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## GPU 基础配置
    parser.add_argument("--gpu", type=str, default='A100a', help="GPU type (A100a, H100)")
    parser.add_argument("--ngpu", type=int, default=8, help="number of GPUs")
    parser.add_argument("--gmemcap", type=int, default=60, help="per-GPU memory capacity (GB)")

    ## PAPI 硬件约束
    parser.add_argument("--powerlimit", action='store_true', help="Enable PIM power constraint")
    parser.add_argument("--ffopt", action='store_true', help="Enable feedforward parallel optimization")
    parser.add_argument("--pipeopt", action='store_true', help="Enable pipeline optimization")

    ## LLM 负载配置
    parser.add_argument("--model", type=str, default='GPT-175B', help="model list: GPT-175B, LLAMA-65B, MT-530B, OPT-66B")
    parser.add_argument("--word", type=int, default=2, help="Precision: 1(INT8), 2(FP16)")
    parser.add_argument("--lin", type=int, default=2048, help="Prompt length")
    parser.add_argument("--lout", type=int, default=128, help="Output tokens")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")

    args = parser.parse_args()

    # GPU 选择
    gpu_device = GPUType.H100 if args.gpu == 'H100' else GPUType.A100a
    
    # 打印运行信息
    print("Initializing PAPI System...")
    print("Model: {}, Batch: {}, Lin: {}, Lout: {}".format(args.model, args.batch, args.lin, args.lout))

    num_gpu = args.ngpu
    gmem_cap = args.gmemcap * 1024 * 1024 * 1024
    output_path = "output_papi.csv"

    # 数据类型
    dtype = DataType.W16A16 if args.word == 2 else DataType.W8A8
    modelinfos = make_model_config(args.model, dtype)

    # 配置设备
    xpu_config = make_xpu_config(gpu_device, num_gpu=num_gpu, mem_cap=gmem_cap)
    
    system = System(xpu_config['GPU'], modelinfos)

    fc_pim_config = make_pim_config(PIMType.FC, 
                                    InterfaceType.NVLINK4, 
                                    power_constraint=args.powerlimit)
    
    attn_pim_config = make_pim_config(PIMType.ATTN, 
                                      InterfaceType.PCIE5, 
                                      power_constraint=args.powerlimit)

    system.set_papi_accelerators(modelinfos, fc_pim_config, attn_pim_config)

    # 执行仿真
    run(system,
        args.batch,
        args.lin,
        args.lout,
        pipe=args.pipeopt,
        parallel=args.ffopt,
        output_file=output_path,
        power_constraint=args.powerlimit)

if __name__ == "__main__":
    main()