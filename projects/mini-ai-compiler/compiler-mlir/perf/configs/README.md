# Triton 配置目录说明

`perf/configs/` 用于保存 Triton 性能实验的配置文件。

当前建议放入：

- A10 设备上的默认配置
- 参数 sweep 范围
- 每一轮 profile 固定要观察的候选配置

推荐文件命名：

- `triton_linear_relu_a10.json`
- `triton_matmul_a10.json`

建议配置项包括：

- `device`
- `dtype`
- `default`
  - `BLOCK_M`
  - `BLOCK_N`
  - `BLOCK_K`
  - `num_warps`
  - `num_stages`
- `sweep`
  - `BLOCK_M`
  - `BLOCK_N`
  - `BLOCK_K`
  - `num_warps`
  - `num_stages`
- `profile_target`

当前目录已经落地了第一版示例配置：

- `triton_linear_relu_a10.json`

后续扩展到 `matmul` 时再补相应配置文件。
