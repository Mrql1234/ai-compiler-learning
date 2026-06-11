# Triton 迭代记录目录说明

`perf/notes/` 用于保存 Triton 性能迭代的文字记录。

建议每一轮实验都至少写清楚：

- 改了什么参数或写法
- `kernel_ms` 怎么变化
- Nsight 指标怎么变化
- 对变化的原因判断
- 下一轮准备修改什么

推荐文件命名：

- `triton_linear_relu_iterations.md`
- `triton_matmul_iterations.md`

这个目录的作用不是替代 benchmark JSON，而是补上“为什么这么改”的解释层。
