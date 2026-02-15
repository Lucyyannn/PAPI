import pandas as pd
import numpy as np
import copy
from .type import *
from .model import *
from .devices import *
from .config import *
RAMPATH = "./ramulator2"
RAMLOG = "./ramulator.out"

LOG_PRINT = True


class System:

    def __init__(self,
                 gpu_config,
                 papi_alpha=3,
                 modelinfos=None):
        self.scaling_factor = SCALING_FACTOR
        self.hetero_name = DeviceType.NONE

        self.GPU = xPU(DeviceType.GPU, gpu_config, self.scaling_factor)
        self.fc_pim = None
        self.attn_pim = None
        
        self.model_set = 0
        if modelinfos is not None:
            self.model = Transformer(modelinfos,
                                     tensor_parallel=self.GPU.num_xpu)
            self.model_set = 1
        self.speculation_length = 1
        self.papi_alpha = papi_alpha # 调度阈值 alpha

    def set_model(self, modelinfos):
        self.model = Transformer(modelinfos, tensor_parallel=self.GPU.num_xpu)
        self.model_set = 1

    def set_papi_accelerators(self, modelinfos, fc_config, attn_config):
        self.hetero_name = DeviceType.PIM
        
        # 初始化 FC-PIM  
        fc_ramulator = FCRamulator(modelinfos, "ramulator2", "ramulator_fc.out")
        self.fc_pim = PIM(fc_config, self.scaling_factor, fc_ramulator)
        
        # 初始化 ATTN-PIM  
        attn_ramulator = AttnRamulator(modelinfos, "ramulator2", "ramulator_attn.out")
        self.attn_pim = PIM(attn_config, self.scaling_factor, attn_ramulator)


    def simulate(self,
                 batch_size,
                 perfs=None,
                 pipe=False,
                 parallel_ff=False,
                 power_constraint=True,
                 num_reqs=0,
                 lin=2048,
                 lout=128):

        def _log_print(layer, stage_name, current_rlp, exec_time):
            if LOG_PRINT :
                tflops = layer.get_flops() / exec_time / 1e12 if exec_time > 0 else 0
                
                print("[{}] Step_BS: {}, Layer: {:<10},TFLOPS: {:.4f}, Bound: {:<8}".format(
                    stage_name, current_rlp, layer.name, tflops, layer.bound))
       
        # 加载 Dolly 数据集信息
        try:
            dolly_df = pd.read_csv("dolly.csv")
            # 随机采样一个 Batch 的请求
            if len(dolly_df) >= batch_size:
                workload = dolly_df.sample(n=batch_size).to_dict('records')
            else:
                workload = dolly_df.to_dict('records')
                batch_size = len(workload)
        except Exception as e:
            print(f"Warning: Failed to load dolly.csv ({e})")
            workload = [{'lin': lin, 'lout': lout} for _ in range(batch_size)]

        # 初始化统计变量
        max_lout = max(job['lout'] for job in workload)
        avg_lin = int(np.mean([job['lin'] for job in workload]))
        
        # 构建模型
        self.model.build(batch_size, avg_lin, max_lout, attn_on_hetero=True)
        
        s_decoder = self.model.sum_decoder
        g_decoder = self.model.gen_decoder[0] # 获取单层 Gen 的结构

        # 性能统计容器
        total_time = 0
        total_energy = [0.0] * 6 # [mem, l2, l1, reg, alu, comm]
        
        s_perf = {'all': 0, 'matmul': 0, 'fc': 0, 'comm': 0, 'softmax': 0, 'act': 0, 'norm': 0}
        g_perf = {'all': 0, 'matmul': 0, 'fc': 0, 'comm': 0, 'etc': 0, 'qkv': 0, 'prj': 0, 'ff': 0, 'g2g': 0, 'x2g': 0, 'softmax': 0, 'act': 0, 'norm': 0}
        gen_energies_detail = {LayerType.FC: {'mem': 0, 'comp': 0, 'comm': 0},
                               LayerType.MATMUL: {'mem': 0, 'comp': 0, 'comm': 0},
                               LayerType.SOFTMAX: {'mem': 0, 'comp': 0, 'comm': 0},
                               LayerType.ACT: {'mem': 0, 'comp': 0, 'comm': 0},
                               LayerType.NORM: {'mem': 0, 'comp': 0, 'comm': 0}}


        # 1.Sum 阶段，默认使用GPU
        for layer in s_decoder:
            exec_time, energy = self.GPU.get_time_and_energy(layer)
            
            # 更新统计
            s_perf['all'] += exec_time
            if layer.type == LayerType.FC: s_perf['fc'] += exec_time
            elif layer.type == LayerType.MATMUL: s_perf['matmul'] += exec_time
            elif layer.type == LayerType.SOFTMAX: s_perf['softmax'] += exec_time
            elif layer.type == LayerType.G2G: s_perf['comm'] += exec_time
            
            total_time += exec_time
            for i in range(6): total_energy[i] += energy[i]

            _log_print(layer, 'SUM', batch_size, exec_time) 

        # 2.Gen阶段，动态调度
        # 模拟逐 token 生成，直到 batch 中所有请求完成
        for step in range(max_lout):
            # step1. 计算当前活跃的请求数 (RLP)
            active_requests = [job for job in workload if job['lout'] > step]
            current_rlp = len(active_requests)
            if current_rlp == 0: break

            # step2. 计算 AI = RLP * TLP
            current_ai = current_rlp * self.speculation_length
            
            # step3. 比较，选择 FC 层的执行设备
            fc_bound = "compute"
            if current_ai > self.papi_alpha:
                fc_device = self.GPU
            else:
                fc_device = self.fc_pim
                fc_bound = "memory"
            
            current_tp = self.GPU.num_xpu if fc_device == self.GPU else 1 #若FC运行在GPU上，需多GPU并行

            # 模拟各层执行
            for layer in g_decoder:
                # 动态调整层大小以匹配当前 RLP
                layer = copy.deepcopy(layer)
                # 更新layer相关参数来反映当前的 batch_size、设备
                if layer.type == LayerType.FC:
                    layer.m = current_rlp
                else:
                    layer.numOp = int((self.model.num_heads / self.GPU.num_xpu) * current_rlp)

                # 算子分发
                if layer.type in [LayerType.MATMUL, LayerType.SOFTMAX]:
                    # 1.Attention 永远在 Attn-PIM (1P2B)
                    layer.bound = "memory"
                    layer.n = lin + step # 这里的 n 随 step 增长
                    exec_time, energy = self.attn_pim.get_time_and_energy(layer)
 
                elif layer.type == LayerType.FC:
                    layer.bound = fc_bound
                    # 根据分布节点数重新计算每个节点上的形状 （仅适用于GPT3，不适用于Llama）
                    if layer.name in ['qkv', 'ff1']: 
                        layer.n = int(layer.n / current_tp)
                    elif layer.name in ['proj', 'ff2']:
                        layer.k = int(layer.k / current_tp)
                    # 2.FC 动态调度
                    exec_time, energy = fc_device.get_time_and_energy(layer)

                elif layer.type == LayerType.X2G:
                    # 3.KV 传输通过接口带宽
                    exec_time, energy = self.attn_pim.get_time_and_energy(layer)

                else:
                    # 4.其它留在 GPU
                    exec_time, energy = self.GPU.get_time_and_energy(layer)

                # 更新统计信息
                g_perf['all'] += exec_time
                if layer.type == LayerType.FC:
                    g_perf['fc'] += exec_time
                    if 'ff' in layer.name: g_perf['ff'] += exec_time
                    elif 'qkv' in layer.name: g_perf['qkv'] += exec_time
                    elif 'proj' in layer.name: g_perf['prj'] += exec_time
                elif layer.type == LayerType.MATMUL: g_perf['matmul'] += exec_time
                elif layer.type == LayerType.SOFTMAX: g_perf['softmax'] += exec_time
                elif layer.type in [LayerType.G2G, LayerType.X2G]: g_perf['comm'] += exec_time
                
                if layer.type in gen_energies_detail:
                    gen_energies_detail[layer.type]['mem'] += energy[0]
                    gen_energies_detail[layer.type]['comp'] += sum(energy[1:5])
                    gen_energies_detail[layer.type]['comm'] += energy[5]


                _log_print(layer, 'GEN', current_rlp, exec_time)
                

        # 3. 数据汇总与 Scaling
        g_perf = {k: v / max_lout for k, v in g_perf.items()}
        
        # 构造最终打印格式（严格对应 write_csv 中的 col_name）
        perf_output = list(s_perf.values()) + list(g_perf.values())
        perf_output = [t * self.model.ndec * 1000 for t in perf_output] # ms 转换

        # 能量汇总
        final_energies = [
            sum(gen_energies_detail[t]['mem'] + gen_energies_detail[t]['comp'] + gen_energies_detail[t]['comm'] for t in gen_energies_detail), # total
            sum(gen_energies_detail[t]['mem'] for t in gen_energies_detail),
            0, 0, 0, # L2, L1, Reg (简化)
            sum(gen_energies_detail[t]['comp'] for t in gen_energies_detail), # ALU
            gen_energies_detail[LayerType.FC]['mem'], gen_energies_detail[LayerType.FC]['comp'],
            gen_energies_detail[LayerType.MATMUL]['mem'], gen_energies_detail[LayerType.MATMUL]['comp'],
            gen_energies_detail[LayerType.ACT]['mem'], gen_energies_detail[LayerType.ACT]['comp'],
            sum(gen_energies_detail[t]['comm'] for t in gen_energies_detail) # total comm
        ]

        final_energies = [e * self.model.ndec / 1000 / max_lout for e in final_energies]

        perf_output = list(s_perf.values()) + list(g_perf.values())
        perf_output = [t * self.model.ndec * 1000 for t in perf_output] # ms
        gen_latency_ms = perf_output[len(s_perf)] # 对应 g_perf['all'] 的位置
        throughput = batch_size / (gen_latency_ms / 1000) # tokens/s

        print(f"    [PAPI] Dolly Workload: Batch={batch_size}, Avg_Lin={avg_lin}, Max_Lout={max_lout}")
        print(f"    Throughput: {throughput:.2f} tokens/s, Latency: {gen_latency_ms:.2f}ms")

        # 构造 Tag 和 Config 信息用于 CSV
        tag = [self.model.name, self.model.dtype.name, "GPU", 0, 0, 0] # 占位
        config = ["PAPI_DYNAMIC", self.GPU.num_xpu, pipe, parallel_ff, power_constraint, 
                  0, avg_lin, max_lout, batch_size, 0, 0, 0]

        if perfs is not None:
            perfs.append([tag, config, perf_output, final_energies])

    # 未改，未用到
    def get_required_mem_capacity(self, batch_size, lin, lout):
        ndec = self.model.ndec
        hdim = self.model.hdim
        nhead = self.model.num_heads
        ff_scale = self.model.ff_scale
        w_byte = 2 if self.model.dtype in [DataType.W16A16, DataType.W16A8
                                          ] else 1
        a_byte = 2 if self.model.dtype in [DataType.W16A16, DataType.W8A16
                                          ] else 1
        l = lin + lout - 1

        if 'LLAMA' in self.model.name:
            weight_memory = ndec * hdim * (2 * hdim + 2 * (hdim) +
                                           3 * ff_scale * hdim) * w_byte
        else:
            weight_memory = ndec * hdim * (2 * hdim + 2 * (hdim) +
                                           2 * ff_scale * hdim) * w_byte

        temp_memory = max((hdim + l * nhead) * a_byte, hdim * 2 * a_byte,
                          l * nhead * 2 * a_byte,
                          (ff_scale * hdim + hdim) * a_byte) + l * nhead
        kv_memory = ndec * 2 * l * (hdim) * a_byte

        return weight_memory, kv_memory * batch_size, temp_memory * batch_size

