# PAPI复现说明

## 一、FC-PIM和Attn-PIM的实现
AttAcc基于Ramulator2实现了1P1B的HBM3-PIM模块，专门用于加速Attention计算。PAPI在此基础上修改，得到了4P1B的FC-PIM和1P2B的Attn-PIM。

由于此方法对于kernel在PIM上的任务划分、数据布局等方面都有特定的要求，因此，这部分复现包含两个核心 ———— **Ramulator2仿真器的修改**和**trace生成脚本的设计**。其中后者是最难的部分。

| Interface         | FC-PIM Implementation     | Attn-PIM Implementation  |
| ----------------- | ------------------------- |------------------------- |
| Frontend          | `AttnPIMLoadStoreTrace`   |`FCPIMLoadStoreTrace`     |
| Memory System     | `AttnPIMDRAMSystem`       |`FCPIMDRAMSystem`         |
| DRAM Device       | `Attn-PIM`                |`FC-PIM`                  |
| Address Mapper    | `AttnPIMMap`              |`FCPIMMap`                |
| DRAM Controller   | `AttnPIMController`       |`FCPIMController`         |
| Request Scheduler | `PIM`                     |`PIM`                     |
| Refresh Manager   | `AllBankRefreshHBM3`      |`AllBankRefreshHBM3`      |
| Plugin            | `HBM3TraceRecorder`       |`HBM3TraceRecorder`       |

> 表1 FC-PIM和Attn-PIM在Ramulator2中各接口的实现
### （一）FC-PIM


### （二）Attn-PIM


## 二、PAPI顶层模块调度的实现