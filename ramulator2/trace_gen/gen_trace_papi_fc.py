import argparse
import math

model = "gpt-3-175B"

# PAPI FC-PIM 硬件配置
n_pu_per_bank = 4  # 4P1B
n_hbm = 1
n_channel = 16 
n_pch = 2
n_rank = 2
n_bg = 3          
n_bank = 4
n_row = pow(2, 14)
n_col = pow(2, 5)
prefetch_size = 32 # Byte，访存粒度
data_size_byte = 2 # FP16 ， 2 byte
n_mac = 16         # 32B / 2B = 16 (一个PU一次处理16个FP16)

# Granularity size
HBM_GS = {}
HBM_GS['col']     = prefetch_size
HBM_GS['row']     = n_col * HBM_GS['col']
HBM_GS['ba']      = n_row * HBM_GS['row'] 
HBM_GS['bg']      = n_bank * HBM_GS['ba'] 
HBM_GS['rank']    = n_bg * HBM_GS['bg'] 
HBM_GS['pch']     = n_rank * HBM_GS['rank'] 
HBM_GS['ch']      = n_pch * HBM_GS['pch']
HBM_GS['hbm']     = n_channel * HBM_GS['ch']

## --------------------------------------  HBM memory space ----------------------------------------------------##
## ------|  legacy CH  |  pCH  |  rank  | BG | BA |  row index  |  column index  |  access granularity  |------ ##
## bits  |     4       |   1   |   1    | 2  | 2  |     14      |        5       |          5           |       ##

## ----------------------------  Commands -------------------------------##
## MACAB: 8tCK (tCCDLx 2) ； 连续计算 batch_size = 8
##  WRGB: 4tCK (write to SRAM not DRAM)
##  RFGB: 4tCK

cmd_wr_gb = []  # 写入输入向量 
cmd_mac   = []  # GEMV计算 
cmd_rf_gb = []  # 读出计算结果 

def cmd_list_reset():
    global cmd_wr_gb, cmd_mac, cmd_rf_gb
    cmd_wr_gb = []
    cmd_mac   = []
    cmd_rf_gb = []

def FC_Layer(h_in, h_out, batch_size, addr_offset):
    '''
    (batch * h_in) * (h_in * h_out) = (batch * h_out)
    '''
    # channel级划分 → smaller 2D matrix ; [2,8]/[8,2]
    n_channel_col = 8 if h_out > h_in else 2
    n_channel_row = n_channel//n_channel_col

    # 列划分
    col_parallel_units = n_channel_col * n_pch * n_rank * n_bg 
    cols_per_unit = math.ceil(h_out/col_parallel_units)
    # 行划分
    row_parallel_units = n_channel_row * n_bank * n_pu_per_bank
    rows_per_unit = math.ceil(h_in / row_parallel_units)
    # 一个PU里的Buffer大小（假设）
    pu_space = batch_size * math.ceil(rows_per_unit/n_mac) * prefetch_size


    # WRGB: 将输入矩阵 [Batch, h_in] 写入 GEMV Buffer
    for b in range(batch_size):
        for d_step in range(math.ceil(rows_per_unit/n_mac)):#对每个PU写的次数
            for row_pu_idx in range(row_parallel_units):#对所有PU
                # 计算物理位置
                lch_row = row_pu_idx // (n_bank * n_pu_per_bank)
                ba_idx = (row_pu_idx % (n_bank * n_pu_per_bank)) // n_pu_per_bank
                pu_idx = row_pu_idx % n_pu_per_bank
                
                for lch_col in range(n_channel_col):
                    # 计算实际 Channel ID
                    lch = lch_row * n_channel_col + lch_col
                    # 地址计算：定位到Bank + 区分PU + 区分GEMV Buffer中哪个batch的行和位置 
                    addr = addr_offset + lch * HBM_GS['ch'] + ba_idx * HBM_GS['ba'] + \
                            + pu_idx * pu_space + (b * math.ceil(rows_per_unit/n_mac) + d_step) * prefetch_size
                    hex_addr = hex(addr)[2:]
                    cmd_wr_gb.append("PIM_WR_GB 0x{0:0>8}".format(hex_addr))

    # MACAB 
    # 每个计算发给哪个PU未体现 / 4PU体现在多batch计算延迟的掩盖
    for c_idx in range(cols_per_unit):#逐列
        cmd_mac.append([])
        for r_idx in range(math.ceil(rows_per_unit / n_mac)):#每16个（32B）一组，分别在PU01230123...中计算
            idx = r_idx + c_idx * math.ceil(rows_per_unit / n_mac) #在bank中的总序号 
            for lch in range(n_channel):
                for b in range(batch_size):#对多个请求分别发送一条命令；在Ramulator2中直接延时1个cycle表示计算
                    addr = addr_offset + lch * HBM_GS['ch'] + idx * HBM_GS['col']
                    hex_addr = hex(addr)[2:]
                    cmd_mac[-1].append("PIM_MAC_AB 0x{0:0>8}".format(hex_addr))
        # RFGB
        # 每得到16个结果元素就运输出去
        if c_idx % 16 == 15 or c_idx == cols_per_unit - 1:
            cmd_rf_gb.append([])
            for b in range(batch_size):
                for bg_idx in range(n_bg):   
                    for rank in range(n_rank):
                        #for pch in range(n_pch):
                        
                        for lch in range(n_channel):
                            addr = addr_offset + lch * HBM_GS['ch'] + \
                                    rank * HBM_GS['rank'] + bg_idx * HBM_GS['bg'] #精确到BG，因为累加器在BG
                            hex_addr = hex(addr)[2:]
                            cmd_rf_gb[-1].append("PIM_RF_GB 0x{0:0>8}".format(hex_addr))


def run_fc_layer(h_in, h_out, batch_size, trace_file_name):
    '''
    trace生成: wrgb , [mac,rfgb]交错
    '''
    cmd_list_reset()
    
    # 模拟一层计算
    addr_offset = 0 #每个decoder层的权重在bank中存储的起始偏移
    FC_Layer(h_in, h_out, batch_size, addr_offset)

    # 生成 Barrier 指令 ：所有channel的同步屏障
    barrier = []
    for lch in range(n_channel):
        addr = lch * HBM_GS['ch']
        hex_addr = hex(addr)[2:]
        barrier.append("PIM_BARRIER 0x{0:0>8}".format(hex_addr))

    total_cmd = []

    # WRGB
    total_cmd.extend(cmd_wr_gb)
    total_cmd.extend(barrier)#保证数据全部写入后再计算
    # [MAC，RFGB]
    num_rf_chunks = len(cmd_rf_gb)
    for j in range(num_rf_chunks + 1):#[0,num_rf_chunk-1]计算,[1,num_rf_chunk]搬运
        # (1)计算第j个chunk（每个chunk包含16列）
        if j < num_rf_chunks:
            for k in range(16):
                c_idx = j * 16 + k
                if c_idx >= len(cmd_mac):
                    break
                total_cmd.extend(cmd_mac[c_idx])
        
        # (2)搬运第 j-1 个 chunk 的结果
        if j > 0:
            total_cmd.extend(cmd_rf_gb[j-1])
            
        # (3)每个流水线阶段后加一个 Barrier 确保同步
        if j < num_rf_chunks:
            total_cmd.extend(barrier)


    trace_file = open(trace_file_name, 'w')
    for cmd in total_cmd:
        trace_file.write(cmd + "\n")

    trace_file.close()
    print(f"Trace 已成功保存至: {trace_file_name}")

# =======================================================
# 主函数
# =======================================================
def main():
    parser = argparse.ArgumentParser(description="PAPI FC-PIM Trace Generator (4P1B)")
    parser.add_argument("-hin", "--h_in", type=int, default=12288, help="隐藏层输入维度")
    parser.add_argument("-hout", "--h_out", type=int, default=12288, help="隐藏层输出维度")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="batch_size 大小")
    parser.add_argument("-o", "--output", type=str, default="papi_fc.trace", help="输出trace文件路径")

    args = parser.parse_args()

    run_fc_layer(args.h_in, args.h_out, args.batch_size, args.output)

if __name__ == "__main__":
    main()