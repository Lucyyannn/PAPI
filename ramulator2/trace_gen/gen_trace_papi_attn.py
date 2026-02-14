import argparse
import math
import copy
import numpy as np

model = "gpt-3-175B"

dhead = 128
max_L = 2048
data_size = 16 # FP 16

n_attacc = 8
max_n_hbm = 8
n_hbm = 5
n_channel = 16
n_pch = 2
n_rank = 2
n_bank = 4
n_bg = 4
n_row = pow(2, 14)
n_col = pow(2, 5)
prefetch_size = 32 # byte
n_mac = 16 # 因为FP16是16bit，2字节。prefetch_size=32 byte，一次访存取出16个数,因此一个PU需要一次进行16个运算


# Granularity size
HBM_GS = {}
HBM_GS['col']     = prefetch_size
HBM_GS['row']     = n_col * HBM_GS['col']
HBM_GS['ba']      = n_row * HBM_GS['row'] 
HBM_GS['bg']      = n_bank * HBM_GS['ba'] 
HBM_GS['rank']     = n_bg * HBM_GS['bg'] 
HBM_GS['pch']     = n_rank * HBM_GS['rank'] 
HBM_GS['ch']      = n_pch * HBM_GS['pch']
HBM_GS['hbm']     = n_channel * HBM_GS['ch']
HBM_GS['attacc']  = max_n_hbm * HBM_GS['hbm']


## --------------------------------------  HBM memory space ----------------------------------------------------##
## ------|  legacy CH  |  pCH  |  rank  | BG | BA |  row index  |  column index  |  access granularity  |------ ##
## bits  |     4       |   1   |   1    | 2  | 2  |     14      |        5       |          5           |       ##

## ----------------------------  Commands -------------------------------##
## MACAB: 8tCK (tCCDLx 2)
##  WRGB: 4tCK (write to SRAM not DRAM)
##  MVSB: 4tCK
##  MVGB: 4tCK
##  SFM: 16tCK (for L = 256)

cmd_score_wrgb   = []
cmd_score_mac    = []
cmd_score_mvsb   = []
cmd_sfm          = []
cmd_context_mvgb  = []
cmd_context_mac  = []
cmd_context_mvsb = []

valid_channels = []

def cmd_list_reset():
  cmd_score_wrgb   = []
  cmd_score_mac    = []
  cmd_score_mvsb   = []
  cmd_sfm          = []
  cmd_context_mvgb = []
  cmd_context_mac  = []
  cmd_context_mvsb = []

  valid_channel = []

def Attention(L, key_addr, val_addr, itr, valid_channel = n_channel):
  cmd_score_wrgb.append([])
  cmd_score_mac.append([])
  cmd_score_mvsb.append([])
  cmd_sfm.append([])
  cmd_context_mvgb.append([])
  cmd_context_mac.append([])
  cmd_context_mvsb.append([])

  valid_channels.append(valid_channel)

  # 把 Q 向量写入 PIM 的 GEMV buffer（GB）  
  def score_cpvec(addr_offset, L):
    # Data broadcasting for pch, rank, bg, and ba
    for ba_idx in range(0,n_bank,2): # number of partitions
      for col_idx in range(math.ceil(dhead / (n_bank/2) / n_mac)):
        if col_idx %2 ==0:
          for lch in range(math.ceil(valid_channel)):
            addr1 = addr_offset + lch * HBM_GS['ch'] + ba_idx * HBM_GS['ba'] + col_idx//2 #显式给出对每个channel的rank0&1,bg0的所有bank的PU写入的请求；其余靠广播
            hex_addr1 = hex(addr1)[2:]
            cmd_score_wrgb[itr].append("PIM_WR_GB 0x{0:0>8}".format(hex_addr1))
        else:
          for lch in range(math.ceil(valid_channel)):
            addr2 = addr_offset + lch * HBM_GS['ch'] + (ba_idx+1) * HBM_GS['ba'] + col_idx//2 #显式给出对每个channel的rank0&1,bg0的所有bank的PU写入的请求；其余靠广播
            hex_addr2 = hex(addr2)[2:]
            cmd_score_wrgb[itr].append("PIM_WR_GB 0x{0:0>8}".format(hex_addr2))

  def score_mac(addr_offset, L):
    # [1,dhead] * [dhead,L] = [1,L]
    for n_idx in range(math.ceil(L / n_pch / n_rank / n_bg)):#逐列计算
      cmd_score_mac[itr].append([])
      for k_idx in range(math.ceil(dhead / (n_bank/2) / n_mac / 2)): #对每列，16行为一组执行MAC
        idx = k_idx + n_idx * math.ceil(dhead / (n_bank/2) / n_mac / 2) #在单个bank中的总序号   

        # All bank command 
        for lch in range(math.ceil(valid_channel)):# 只要是不同channel，就是并行执行的，这是controller并行导致
          addr = addr_offset + lch * HBM_GS['ch'] + idx * HBM_GS['col']
          hex_addr = hex(addr)[2:]
          cmd_score_mac[itr][-1].append("PIM_MAC_AB 0x{0:0>8}".format(hex_addr))
         ## parallelization 
         # #不过由于power限制，能同时运行的GEMV unit数为18 per pCH。
         # 所以ACT/PRE的时间可以被其它没有同时执行的bank的read时间掩盖

      ## MVSB command 每16列（计算结果是长16个数的vector，32B）打包一次
      if n_idx % 16 == 15 or n_idx == math.ceil(L / n_pch / n_rank / n_bg) - 1:
        cmd_score_mvsb[itr].append([])
        for bg_idx in range(n_bg):   
          for rank in range(n_rank):
            for lch in range(math.ceil(valid_channel)):
              bank_addr = addr_offset + lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + \
                          bg_idx * HBM_GS['bg'] #精确到bg是因为累加器是BG级的
              hex_addr = hex(bank_addr)[2:]
              cmd_score_mvsb[itr][-1].append("PIM_MV_SB 0x{0:0>8}".format(hex_addr))

  # 将SOFTMAX计算好的P移动到GEMV Buffer
  def context_cpvec(addr_offset, L): 
    # Data broadcasting for bg and ba
    for rank in range(n_rank):
      for bg_idx in range(n_bg):
        for col_idx in range(math.ceil(L / (n_pch * n_rank * n_bg * n_mac))):#对P，每16个元素（32B）一起传输
          # number of columns of partition = L / (R parallel units)
            for lch in range(math.ceil(valid_channel)):
              addr = addr_offset + lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + \
                     bg_idx * HBM_GS['bg'] + col_idx   #分发给每个bg的bank0，然后由bank0广播给bank1~3
              hex_addr = hex(addr)[2:]
              cmd_context_mvgb[itr].append("PIM_MV_GB 0x{0:0>8}".format(hex_addr))

  def context_mac(addr_offset, L):
    # [1,L] * [L,dhead] = [1,dhead]
    for n_idx in range(math.ceil(dhead / ((n_bank/2) * n_mac))):#每个PU要处理的列数
      cmd_context_mac[itr].append([])
      for k_idx in range(math.ceil(L / (n_pch * n_rank * n_bg)/2)): #bank0与bank1交错存储[1,16]的数据。相当于每个bank要存储的容量不变，但矩阵行数/2，列数*2
        idx = k_idx + n_idx * math.ceil(L / (n_pch * n_rank * n_bg /2))
        for lch in range(math.ceil(valid_channel)):
          addr = addr_offset + lch * HBM_GS['ch'] + idx * HBM_GS['col'] 
          hex_addr = hex(addr)[2:]
          cmd_context_mac[itr][-1].append("PIM_MAC_AB 0x{0:0>8}".format(hex_addr))

      ## parallelization. Generate 16 elements per n_idx，
      cmd_context_mvsb[itr].append([])
      for ba_idx in range(0,n_bank,2):#从每个PU里读数据
        for rank in range(n_rank):
          for lch in range(math.ceil(valid_channel)):
            bank_addr = addr_offset + lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + \
                        ba_idx * HBM_GS['ba'] 
            hex_addr = hex(bank_addr)[2:]
            cmd_context_mvsb[itr][-1].append("PIM_MV_SB 0x{0:0>8}".format(hex_addr))

  def softmax(L):
    for lch in range(math.ceil(valid_channel)):
      addr = lch * HBM_GS['ch'] 
      hex_addr = hex(addr)[2:]
      cmd_sfm[itr].append("PIM_SFM 0x{0:0>8}".format(hex_addr))

  score_cpvec(key_addr, L)

  score_mac(key_addr, L)

  softmax(L)

  context_cpvec(val_addr, L)

  context_mac(val_addr, L)


# n_head and n_req = n_req per a HBM 
def run_attention(dhead, n_head_per_hbm, L, trace_file_name):
  partition_size = math.ceil(max_L * dhead / (n_pch * n_rank * n_bg * (n_bank/2))) 
  head_offset = math.ceil(partition_size / 2)
  v_offset = pow(2, 23) # bank的一半空间，上一半给K们，下一半给V们
  

  cmd_list_reset()
  ##-- Generate Commands --##
  num_itr = math.ceil(n_head_per_hbm / (n_channel)) #每个channel的head数
  for itr in range(num_itr):#对一个channel的每个head
    remainder = 0
    if (n_head_per_hbm / ((itr+1) * n_channel) < 1):#说明当前这一轮不需要16个channel全启动
      remainder = n_head_per_hbm % n_channel # 还剩4个head，那么启动4个channel，每ch一个head
    key_addr = itr * head_offset # 当前head在bank中的起始偏移
    val_addr = key_addr + v_offset 

    if remainder == 0:
      Attention(L, key_addr, val_addr, itr)
    else:
      Attention(L, key_addr, val_addr, itr, remainder)


  ##-- Ovelapping Commands --##
  barrier = []
  for lch in range(n_channel):
    addr = lch * HBM_GS['ch']
    hex_addr = hex(addr)[2:]
    barrier.append("PIM_BARRIER 0x{0:0>8}".format(hex_addr))

  total_cmd = []
  for i in range(0, num_itr -1, 2):#对一个channel里，每两个head一起遍历

    # Head0: Score
      ## WRGB
    total_cmd += cmd_score_wrgb[i]
      ## dummy MAC
    if i == 0:
      for j in range(valid_channels[i]):
        total_cmd.append(cmd_score_mac[i][0][j])#3D分别代表：head号，列号(n_idx),列内序号
      ## BARRIER
    total_cmd += barrier

    length = math.ceil(L/n_pch/n_rank/n_bg/16)
    for j in range(0, length+1):
      ## MAC (Head0)
      if not j == length:
        stride = 16
        for k in range(stride):
          if (j*stride+k) >= len(cmd_score_mac[i]):
            break
          total_cmd += cmd_score_mac[i][j*stride+k]
      ## MVSB (Head0)
      if not j == 0:
        total_cmd += cmd_score_mvsb[i][j-1]
      ## WRGB (Head1)
      if not j == length:
        stride = int(n_bank*math.ceil(dhead /n_bank /n_mac)*math.ceil(valid_channels[i+1])/length)
        for k in range(stride):
          if (j*stride+k) >= len(cmd_score_wrgb[i+1]):
            break
          total_cmd.append(cmd_score_wrgb[i+1][j*stride + k])
      ## BARRIER
      if not j == length:
        total_cmd += barrier

    # Head0: SoftMax, Head1: Score
    length = math.ceil(L/n_pch/n_rank/n_bg/16)
    for j in range(0, length+1):
      ## MAC (Head1)
      if not j == length:
        stride = 16
        for k in range(stride):
          if (j*stride+k) >= len(cmd_score_mac[i+1]):
            break
          total_cmd += cmd_score_mac[i+1][j*stride+k]
      ## MVSB (Head1)
      if not j == 0:
        total_cmd += cmd_score_mvsb[i+1][j-1]
      ## SFM (Head0)
      if j == 0:
        total_cmd += cmd_sfm[i]
      ## MVGB (Head0)
      if not j == length:
        if j >= math.floor(length/2):
          stride = int(n_rank*n_bg*math.ceil(L/(n_pch*n_rank*n_bg*n_mac))*math.ceil(valid_channels[i])/math.ceil(length/2))
          for k in range(stride):
            if ((j-math.floor(length/2))*stride + k) >= len(cmd_context_mvgb[i]):
              break
            total_cmd.append(cmd_context_mvgb[i][(j-math.floor(length/2))*stride + k])
      ## BARRIER
      if not j == length:
        total_cmd += barrier

    # Head0: Context, Head1: Softmax
    length = math.ceil(dhead/n_bank/n_mac)
    for j in range(0, length+1):
      ## MAC (Head0)
      if not j == length:
        total_cmd += cmd_context_mac[i][j]
      ## MVSB (Head0)
      if not j == 0:
        total_cmd += cmd_context_mvsb[i][j-1]
      ## SFM (Head1)
      if j == 0:
        total_cmd += cmd_sfm[i+1]
      ## MVGB (Head1)
      if not j == length:
        if j >= math.floor(length/2):
          stride = int(n_rank*n_bg*math.ceil(L/(n_pch*n_rank*n_bg*n_mac))*math.ceil(valid_channels[i+1])/math.ceil(length/2));
          for k in range(stride):
            if ((j-math.floor(length/2))*stride + k) >= len(cmd_context_mvgb[i+1]):
              break
            total_cmd.append(cmd_context_mvgb[i+1][(j-math.floor(length/2))*stride + k])
      ## BARRIER
      if not j == length:
        total_cmd += barrier

    # Head1: Context
    length = math.ceil(dhead/n_bank/n_mac)
    for j in range(0, length+1):
      ## MAC (Head1)
      if not j == length:
        total_cmd += cmd_context_mac[i+1][j] 
      ## MVSB (Head1)
      if not j == 0:
        total_cmd += cmd_context_mvsb[i+1][j-1] 
      ## BARRIER
      if not j == length:
        total_cmd += barrier


  if num_itr % 2 != 0:
    i = num_itr - 1

    # Score
      ## WRGB
    total_cmd += cmd_score_wrgb[i]
      ## BARRIER
    total_cmd += barrier

    length = math.ceil(L/n_pch/n_rank/n_bg/16)
    for j in range(0, length+1):
      ## MAC
      if not j == length:
        stride = 16
        for k in range(stride):
          if (j*stride+k) >= len(cmd_score_mac[i]):
            break
          total_cmd += cmd_score_mac[i][j*stride+k]
      ## MVSB
      if not j == 0:
        total_cmd += cmd_score_mvsb[i][j-1]
      ## BARRIER
      if not j == length:
        total_cmd += barrier

    # SoftMax
    ## SFM (Head0)
    total_cmd += cmd_sfm[i]
    ## MVGB (Head0)
    total_cmd += cmd_context_mvgb[i]
    ## BARRIER
    total_cmd += barrier

    # Context
    length = math.ceil(dhead/n_bank/n_mac)
    for j in range(0, length+1):
      ## MAC
      if not j == length:
        total_cmd += cmd_context_mac[i][j]
      ## MVSB
      if not j == 0:
        total_cmd += cmd_context_mvsb[i][j-1]
      ## BARRIER
      if not j == length:
        total_cmd += barrier


  trace_file = open(trace_file_name, 'w')
  for cmd in total_cmd:
    trace_file.write(cmd + "\n")

  trace_file.close()

def main():
  global dhead, max_L, data_size, n_mac


  parser = argparse.ArgumentParser(description="Output path and operation infos",
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 
  parser.add_argument("-dh", "--dhead", type=int, default=128, 
                      help="dhead, default= 128")
  parser.add_argument("-nh", "--nhead", type=int, default=64,
                      help="Number of heads, default=64")
  parser.add_argument("-l", "--seqlen", type=int, default=2048,
                      help="Sequence length L, default= 2048")
  parser.add_argument("-maxl", "--maxlen", type=int, default=4096, 
                      help="maximum L, default= 4096")
  parser.add_argument("-db", "--dbyte", type=int, default=2, 
                      help="data type (B), default= 2")
  parser.add_argument("-o", "--output", type=str, default="papi_attn.trace", 
                      help="output path")

  args = parser.parse_args()

  dhead = args.dhead
  max_L = args.maxlen
  L = args.seqlen
  n_head_per_hbm = args.nhead 

  data_size = args.dbyte
  n_mac = int(HBM_GS['col'] / data_size)

  print("------   Make a trace of PAPI Attn-PIM   ------")

  args_dict = vars(args)
  print("All Arguments:")
  for key, value in args_dict.items():
      print(f"     {key}: {value}")
  print("---------------------------------------------------")
  run_attention(dhead, n_head_per_hbm, L, args.output)



if __name__ == "__main__":
  main()
