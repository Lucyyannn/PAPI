# PAPI复现思路

| Interface         | Implementation            |Implementation            |
| ----------------- | ------------------------- |------------------------- |
| Frontend          | `AttnPIMLoadStoreTrace`   |`FCPIMLoadStoreTrace`     |
| Memory System     | `AttnPIMDRAMSystem`       |`FCPIMDRAMSystem`         |
| DRAM Device       | `Attn-PIM`                |`FC-PIM`                  |
| Address Mapper    | `AttnPIMMap`              |`FCPIMMap`                |
| DRAM Controller   | `AttnPIMController`       |`FCPIMController`         |
| Request Scheduler | `PIM`                     |`PIM`                     |
| Refresh Manager   | `AllBankRefreshHBM3`      |`AllBankRefreshHBM3`      |
| Plugin            | `HBM3TraceRecorder`       |`HBM3TraceRecorder`       |
