# Triton sweep summary: triton_linear_relu_f32_m1024_n1024_k1024

| rank | config | kernel_ms median | invoke_ms median | correct | json |
| --- | --- | --- | --- | --- | --- |
| 1 | bm128_bn128_bk32_gm4_w8_s3 | 0.254976 | 0.270839 | yes | /home/ql/code/ai-compiler-learning/projects/mini-ai-compiler/compiler-mlir/perf/runs/triton_iterations/iter_01_tile/115_bm128_bn128_bk32_gm4_w8_s3.json |
| 2 | bm128_bn128_bk32_gm8_w8_s3 | 0.254976 | 0.271124 | yes | /home/ql/code/ai-compiler-learning/projects/mini-ai-compiler/compiler-mlir/perf/runs/triton_iterations/iter_01_tile/124_bm128_bn128_bk32_gm8_w8_s3.json |
| 3 | bm128_bn128_bk32_gm8_w8_s4 | 0.254976 | 0.270976 | yes | /home/ql/code/ai-compiler-learning/projects/mini-ai-compiler/compiler-mlir/perf/runs/triton_iterations/iter_01_tile/125_bm128_bn128_bk32_gm8_w8_s4.json |
| 4 | bm128_bn128_bk32_gm4_w8_s4 | 0.256000 | 0.273154 | yes | /home/ql/code/ai-compiler-learning/projects/mini-ai-compiler/compiler-mlir/perf/runs/triton_iterations/iter_01_tile/116_bm128_bn128_bk32_gm4_w8_s4.json |
| 5 | bm128_bn128_bk64_gm4_w8_s2 | 0.266240 | 0.282094 | yes | /home/ql/code/ai-compiler-learning/projects/mini-ai-compiler/compiler-mlir/perf/runs/triton_iterations/iter_01_tile/132_bm128_bn128_bk64_gm4_w8_s2.json |
| 6 | bm128_bn128_bk64_gm8_w8_s2 | 0.266240 | 0.282238 | yes | /home/ql/code/ai-compiler-learning/projects/mini-ai-compiler/compiler-mlir/perf/runs/triton_iterations/iter_01_tile/141_bm128_bn128_bk64_gm8_w8_s2.json |
| 7 | bm64_bn64_bk32_gm4_w2_s2 | 0.267264 | 0.283027 | yes | /home/ql/code/ai-compiler-learning/projects/mini-ai-compiler/compiler-mlir/perf/runs/triton_iterations/iter_01_tile/000_bm64_bn64_bk32_gm4_w2_s2.json |
| 8 | bm64_bn64_bk32_gm8_w2_s2 | 0.268288 | 0.283609 | yes | /home/ql/code/ai-compiler-learning/projects/mini-ai-compiler/compiler-mlir/perf/runs/triton_iterations/iter_01_tile/009_bm64_bn64_bk32_gm8_w2_s2.json |
| 9 | bm64_bn128_bk64_gm4_w4_s2 | 0.268288 | 0.284331 | yes | /home/ql/code/ai-compiler-learning/projects/mini-ai-compiler/compiler-mlir/perf/runs/triton_iterations/iter_01_tile/057_bm64_bn128_bk64_gm4_w4_s2.json |
| 10 | bm64_bn128_bk64_gm8_w4_s2 | 0.268288 | 0.283945 | yes | /home/ql/code/ai-compiler-learning/projects/mini-ai-compiler/compiler-mlir/perf/runs/triton_iterations/iter_01_tile/066_bm64_bn128_bk64_gm8_w4_s2.json |
