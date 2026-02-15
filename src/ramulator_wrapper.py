import pandas as pd
import subprocess
import math
import os
from src.config import *
from src.model import *
from src.type import *


class AttnRamulator:

    def __init__(self,
                 modelinfos,
                 ramulator_dir,
                 output_log='',
                 fast_mode=False,
                 num_hbm=5):
        self.df = pd.DataFrame()
        self.ramulator_dir = ramulator_dir
        self.output_log = output_log
        if os.path.exists(output_log):
            self.df = pd.read_csv(output_log)
        self.tCK = 0.769  # ns
        self.num_hbm = num_hbm
        self.nhead = modelinfos['num_heads']
        self.dhead = modelinfos['dhead']
        self.fast_mode = fast_mode

    def make_yaml_file(self, yaml_file, file_name):
        trace_path = os.path.join(self.ramulator_dir, file_name + ".trace")

        content = f"""Frontend:
  impl: AttnPIMLoadStoreTrace
  path: {trace_path}
  clock_ratio: 1

  Translation:
    impl: NoTranslation
    max_addr: 2147483648
              

MemorySystem:
  impl: AttnPIMDRAM
  clock_ratio: 1
  DRAM:
    impl: Attn-PIM
    org:
      preset: HBM3_8Gb_2R
      channel: 16
    timing:
      preset: HBM3_5.2Gbps

  Controller:
    impl: Attn-PIM
    Scheduler:
      impl: PIM
    RefreshManager:
      impl: AllBankHBM3
    plugins:

  AddrMapper:
    impl: Attn-PIM
"""
        with open(yaml_file, 'w') as f:
            f.write(content)

    def update_log_file(self, log):
        if self.df.empty:
            if os.path.exists(self.output_log):
                df = pd.read_csv(self.output_log)
            else:
                columns = [
                    'L', 'nhead', 'dhead', 'dbyte',
                    'power_constraint', 'cycle', 'mac', 'softmax', 'mvgb',
                    'mvsb', 'wrgb'
                ]
                df = pd.DataFrame(columns=columns)
        else:
            df = self.df
        if len(df.columns) > 11:
            import pdb
            pdb.set_trace()
        new_df = pd.DataFrame(columns=df.columns)
        new_df.loc[0] = log
        df = pd.concat([df, new_df]).drop_duplicates()
        self.df = df
        self.df.to_csv(self.output_log, index=False)

    #def run_ramulator(self):
    def run_ramulator(self,  l, num_ops_per_hbm, dbyte,
                      yaml_file, file_name):
        trace_file = os.path.join(self.ramulator_dir, file_name + '.trace')

        trace_exc = os.path.join(
            self.ramulator_dir,
            "trace_gen/gen_trace_papi_attn.py")
        trace_args = "--dhead {} --nhead {} --seqlen {} --dbyte {} --output {}".format(
            self.dhead, num_ops_per_hbm, l, dbyte, trace_file)

        gen_trace_cmd = f"python3 {trace_exc} {trace_args}"

        # generate trace
        try:
            os.system(gen_trace_cmd)
        except Exception as e:
            print(f"Error: {e}")

        # run ramulator
        ramulator_file = os.path.join(self.ramulator_dir, "ramulator2")
        run_ramulator_cmd = f"{ramulator_file} -f {yaml_file}"
        try:
            result = subprocess.run(run_ramulator_cmd,
                                    stdout=subprocess.PIPE,
                                    text=True,
                                    shell=True)
            output_lines = result.stdout.strip().split('\n')
            output_list = [line.strip() for line in output_lines]
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            assert 0

        # remove trace
        rm_trace_cmd = f"rm {trace_file}"
        try:
            os.system(rm_trace_cmd)
        except Exception as e:
            print(f"Error: {e}")

        # parsing output
        n_cmds = {"mac": 0, "sfm": 0, "mvgb": 0, "mvsb": 0, "wrgb": 0}
        cycle = 0
        for line in output_list:
            if "mac" in line:
                n_cmds["mac"] += int(line.split()[-1])
            elif "softmax_requests" in line:
                n_cmds["sfm"] += int(line.split()[-1])
            elif "move_to_gemv_buffer" in line:
                n_cmds["mvgb"] += int(line.split()[-1])
            elif "move_to_softmax_buffer" in line:
                n_cmds["mvsb"] += int(line.split()[-1])
            elif "write_to_gemv_buffer" in line:
                n_cmds["wrgb"] += int(line.split()[-1])
            elif "memory_system_cycles" in line:
                cycle += int(line.split()[-1])

        out = [
            cycle, n_cmds["mac"], n_cmds["sfm"], n_cmds["mvgb"], n_cmds["mvsb"],
            n_cmds["wrgb"]
        ]
        return out

    def run(self, layer: Layer, power_constraint=True):
        if os.path.exists(self.ramulator_dir):
            l = layer.n
            dhead = self.dhead
            dbyte = layer.dbyte
            num_ops_per_attacc = layer.numOp
            num_ops_per_hbm = math.ceil(num_ops_per_attacc / self.num_hbm)
            num_ops_group = 1
            if self.fast_mode:
                minimum_heads = 64
                num_ops_group = math.ceil(num_ops_per_hbm / minimum_heads)
                num_ops_per_hbm = minimum_heads

            file_name = "papi_l{}_nattn{}_dhead{}_dbyte{}_pc{}".format(
                l, num_ops_per_hbm, dhead, layer.dbyte, int(power_constraint))
            yaml_file = os.path.join(self.ramulator_dir, file_name + '.yaml')
            self.make_yaml_file(yaml_file, file_name, power_constraint)

            result = self.run_ramulator(l, num_ops_per_hbm,
                                        layer.dbyte, yaml_file, file_name)

            # remove trace
            rm_yaml_cmd = f"rm {yaml_file}"
            try:
                os.system(rm_yaml_cmd)
            except Exception as e:
                print(f"Error: {e}")

            # post processing
            # 32: read granularity
            cycle, mac, sfm, mvgb, mvsb, wrgb = result
            si_io = wrgb * 32  # 256 bit
            tsv_io = (wrgb + mvsb + mvgb) * 32
            giomux_io = (wrgb + mvsb + mvgb) * 32
            bgmux_io = (wrgb + mvsb + mvgb) * 32
            mem_acc = (mac * 32) *2 *2 *4 *4  # pCH * Rank * bank group * bank

            ## update log file
            log = [
                l, num_ops_per_hbm, dhead, dbyte, 
                power_constraint
            ] + result
            self.update_log_file(log)

            ## si, tsv, giomux to bgmux, bgmux to column decoder, bank RD
            traffic = [si_io, tsv_io, giomux_io, bgmux_io, mem_acc]
            traffic = [i * self.num_hbm for i in traffic]
            traffic = [i * num_ops_group for i in traffic]
            exec_time = self.tCK * cycle / 1000 / 1000 / 1000  # ns -> s
            return exec_time, traffic

        else:
            assert 0, "Need to install ramulator"

    def output(self, pim_type: PIMType, layer: Layer, power_constraint=True):
        if self.df.empty:
            self.run(pim_type, layer, power_constraint)

        num_ops_per_attacc = layer.numOp
        num_ops_per_hbm = math.ceil(num_ops_per_attacc / self.num_hbm)
        num_ops_group = 1
        if self.fast_mode:
            minimum_heads = 64
            num_ops_group = math.ceil(num_ops_per_hbm / minimum_heads)
            num_ops_per_hbm = minimum_heads

        l = layer.n
        dhead = layer.k
        dbyte = layer.dbyte
        row = self.df[(self.df['L'] == l) & (self.df['nhead'] == num_ops_per_hbm) & \
                      (self.df['dbyte'] == dbyte) & (self.df['dhead'] == dhead) & \
                      (self.df['power_constraint'] == power_constraint) &  \
                      (self.df['pim_type'] == pim_type.name)]
        if row.empty:
            return self.run(pim_type, layer, power_constraint)

        else:
            cycle = int(row.iloc[0]['cycle'])
            mac = int(row.iloc[0]['mac'])
            softmax = int(row.iloc[0]['softmax'])
            mvgb = int(row.iloc[0]['mvgb'])
            mvsb = int(row.iloc[0]['mvsb'])
            wrgb = int(row.iloc[0]['wrgb'])
            si_io = wrgb * 32  # 256 bit
            tsv_io = (wrgb + mvsb + mvgb) * 32
            giomux_io = (wrgb + mvsb + mvgb) * 32
            bgmux_io = (wrgb + mvsb + mvgb) * 32
            mem_acc = (mac * 32) * 2 * 2 * 4 * 4

            ## si, tsv, giomux to bgmux, bgmux to column decoder, bank RD
            traffic = [si_io, tsv_io, giomux_io, bgmux_io, mem_acc]
            traffic = [i * self.num_hbm for i in traffic]
            traffic = [i * num_ops_group for i in traffic]
            exec_time = self.tCK * cycle / 1000 / 1000 / 1000  # ns -> s
            exec_time *= num_ops_group
            return exec_time, traffic


class FCRamulator:

    def __init__(self,
                 modelinfos,
                 ramulator_dir,
                 output_log='',
                 fast_mode=False,
                 num_hbm=1):
        self.df = pd.DataFrame()
        self.ramulator_dir = ramulator_dir
        self.output_log = output_log
        if os.path.exists(output_log):
            self.df = pd.read_csv(output_log)
        self.tCK = 0.769  # ns
        self.num_hbm = num_hbm
        # 对于 FC 层，modelinfos 仍然包含基础维度，但 run 阶段会从 layer 直接提取
        self.fast_mode = fast_mode

    def make_yaml_file(self, yaml_file, file_name):
        trace_path = os.path.join(self.ramulator_dir, file_name + ".trace")

        content = f"""Frontend:
  impl: FCPIMLoadStoreTrace
  path: {trace_path}
  clock_ratio: 1

  Translation:
    impl: NoTranslation
    max_addr: 2147483648
              

MemorySystem:
  impl: FCPIMDRAM
  clock_ratio: 1
  DRAM:
    impl: FC-PIM
    org:
      preset: HBM3_8Gb_2R
      channel: 16
    timing:
      preset: HBM3_5.2Gbps

  Controller:
    impl: FC-PIM
    Scheduler:
      impl: PIM
    RefreshManager:
      impl: AllBankHBM3
    plugins:

  AddrMapper:
    impl: FC-PIM
"""
        with open(yaml_file, 'w') as f:
            f.write(content)

    def update_log_file(self, log):
        if self.df.empty:
            if os.path.exists(self.output_log):
                df = pd.read_csv(self.output_log)
            else:
                columns = [
                    'hin', 'hout_per_hbm', 'batch', 'dbyte',
                    'power_constraint', 'cycle', 'mac', 'wrgb', 'rfgb'
                ]
                df = pd.DataFrame(columns=columns)
        else:
            df = self.df
            
        new_df = pd.DataFrame(columns=df.columns)
        new_df.loc[0] = log
        df = pd.concat([df, new_df]).drop_duplicates()
        self.df = df
        self.df.to_csv(self.output_log, index=False)

    def run_ramulator(self, h_in, h_out_per_hbm, batch_size, dbyte,
                      yaml_file, file_name):
        trace_file = os.path.join(self.ramulator_dir, file_name + '.trace')
        
        trace_exc = os.path.join(
            self.ramulator_dir,
            "trace_gen/gen_trace_papi_fc.py")
        
        trace_args = "-hin {} -hout {} -b {} -o {}".format(
            h_in, h_out_per_hbm, batch_size, trace_file)

        gen_trace_cmd = f"python3 {trace_exc} {trace_args}"

        try:
            os.system(gen_trace_cmd)
        except Exception as e:
            print(f"Error generating FC trace: {e}")

        ramulator_file = os.path.join(self.ramulator_dir, "ramulator2")
        run_ramulator_cmd = f"{ramulator_file} -f {yaml_file}"
        
        try:
            result = subprocess.run(run_ramulator_cmd,
                                    stdout=subprocess.PIPE,
                                    text=True,
                                    shell=True)
            output_list = [line.strip() for line in result.stdout.strip().split('\n')]
        except subprocess.CalledProcessError as e:
            print(f"Error running FC ramulator: {e}")
            assert 0

        # 清理 trace
        if os.path.exists(trace_file):
            os.system(f"rm {trace_file}")

        # 解析 FC-PIM 专有的输出字段
        n_cmds = {"mac": 0, "wrgb": 0, "rfgb": 0}
        cycle = 0
        for line in output_list:
            if "pim_mac_all_bank" in line:
                n_cmds["mac"] += int(line.split()[-1])
            elif "pim_write_to_gemv_buffer" in line:
                n_cmds["wrgb"] += int(line.split()[-1])
            elif "pim_read_from_gemv_buffer" in line:
                n_cmds["rfgb"] += int(line.split()[-1])
            elif "memory_system_cycles" in line:
                cycle += int(line.split()[-1])

        return [cycle, n_cmds["mac"], n_cmds["wrgb"], n_cmds["rfgb"]]

    def run(self, layer: Layer, power_constraint=True):
        if os.path.exists(self.ramulator_dir):
            h_in = layer.k
            batch_size = layer.m
            h_out_per_hbm = layer.n #h_out_per_hbm = math.ceil(layer.n / self.num_hbm)

            file_name = "papi_fc_hin{}_hout{}_b{}_pc{}".format(
                h_in, h_out_per_hbm, batch_size, int(power_constraint))
            yaml_file = os.path.join(self.ramulator_dir, file_name + '.yaml')
            self.make_yaml_file(yaml_file, file_name)

            result = self.run_ramulator(h_in, h_out_per_hbm, batch_size,
                                        layer.dbyte, yaml_file, file_name)

            if os.path.exists(yaml_file):
                os.system(f"rm {yaml_file}")

            # 解析结果
            cycle, mac, wrgb, rfgb = result
            
            # 数据量计算 
            si_io = wrgb * 32
            tsv_io = (wrgb + rfgb) * 32
            giomux_io = (wrgb + rfgb) * 32
            bgmux_io = (wrgb + rfgb) * 32
            
            # 计算 Memory Access: pCH(2) * Rank(2) * BG(3) * Bank(4)
            mem_acc = (mac * 32) * 2 * 2 * 3 * 4 

            # 更新日志
            log = [h_in, h_out_per_hbm, batch_size, layer.dbyte, power_constraint] + result
            self.update_log_file(log)

            traffic = [si_io, tsv_io, giomux_io, bgmux_io, mem_acc]
            traffic = [i * self.num_hbm  for i in traffic]# 基于单 HBM 的 trace 结果扩展到全系统
            exec_time = (self.tCK * cycle ) / 1e9 # s
            
            return exec_time, traffic
        else:
            assert 0, "Ramulator directory not found"

    def output(self, pim_type: PIMType, layer: Layer, power_constraint=True):
        # 尝试从缓存读取
        h_in = layer.k
        batch_size = layer.m
        h_out_per_hbm = layer.n # math.ceil(layer.n / self.num_hbm)

        if not self.df.empty:
            # 缓存查询条件更新
            row = self.df[(self.df['hin'] == h_in) & 
                          (self.df['hout_per_hbm'] == h_out_per_hbm) & 
                          (self.df['batch'] == batch_size) &
                          (self.df['power_constraint'] == power_constraint)]
            if not row.empty:
                cycle = int(row.iloc[0]['cycle'])
                mac = int(row.iloc[0]['mac'])
                wrgb = int(row.iloc[0]['wrgb'])
                rfgb = int(row.iloc[0]['rfgb'])

                si_io = wrgb * 32
                tsv_io = (wrgb + rfgb) * 32
                giomux_io = (wrgb + rfgb) * 32
                bgmux_io = (wrgb + rfgb) * 32
                mem_acc = (mac * 32) * 2 * 2 * 3 * 4

                traffic = [si_io, tsv_io, giomux_io, bgmux_io, mem_acc]
                traffic = [i * self.num_hbm  for i in traffic]
                exec_time = (self.tCK * cycle ) / 1e9
                return exec_time, traffic

        return self.run(layer, power_constraint)