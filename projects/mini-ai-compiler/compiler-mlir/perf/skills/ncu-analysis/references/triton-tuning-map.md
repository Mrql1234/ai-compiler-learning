# Triton Tuning Map

## Contents

1. Symptom to change mapping
2. Parameter meanings
3. Safe experiment patterns

## 1. Symptom To Change Mapping

### Symptom: high DRAM pressure, weak SM utilization

Try:

- increase tile reuse where reasonable
- test larger `BLOCK_M` or `BLOCK_N`
- improve program ordering
- check whether fusion can reduce extra reads or writes

### Symptom: high register pressure, low occupancy

Try:

- reduce tile size
- reduce `num_warps`
- reduce `num_stages`

### Symptom: long scoreboard or memory-latency stalls

Try:

- increase `num_stages` gradually
- keep tile sizes fixed while testing pipeline depth
- re-check whether occupancy collapses after the change

### Symptom: weak cache locality

Try:

- change `GROUP_M`
- test grouped ordering
- check whether L2 hit rate improves

### Symptom: kernel under-fills the GPU

Try:

- confirm the problem is large enough to be representative
- avoid drawing strong conclusions from tiny smoke cases

## 2. Parameter Meanings

### `BLOCK_M`, `BLOCK_N`, `BLOCK_K`

- define the tile geometry
- change reuse, parallelism, and boundary-mask overhead

### `num_warps`

- sets how many warps collaborate on a Triton program
- affects per-tile parallelism and resource use

### `num_stages`

- controls software pipeline depth
- can improve latency hiding
- can also raise resource pressure

### `GROUP_M`

- changes program ordering
- often affects cache locality more than raw arithmetic structure

## 3. Safe Experiment Patterns

Prefer these patterns:

- fix tile, sweep `num_warps`
- fix tile and `num_warps`, sweep `num_stages`
- fix compute shape, compare 2 nearby `GROUP_M` values

Avoid:

- changing every parameter at once
- reading a single run as final truth
- mixing different shapes while comparing configs
