# LME analysis summary

Each LLM was analysed separately, as specified in the research proposal.
Random effect: intercept for lexical item (`item_id`).
Python reports by-item t-tests of the proposal contrasts (df = 31),
which match random-intercept LME simple effects in this balanced 2×2.
The R script is the canonical lme4 / lmerTest write-up engine.

## Key diagnostics

### local_mismatch_penalty

- BabyLlama: estimate = 2.831, t(31) = 17.69, p = < .001  [supports predicted heuristic/exemption]
- Qwen-2.5-7B: estimate = 3.710, t(31) = 12.74, p = < .001  [supports predicted heuristic/exemption]
- Llama-3.1-8B: estimate = 4.308, t(31) = 17.72, p = < .001  [supports predicted heuristic/exemption]
- Qwen-2.5-72B: estimate = 4.606, t(31) = 12.65, p = < .001  [supports predicted heuristic/exemption]
- Llama-3.1-70B: estimate = 4.264, t(31) = 16.29, p = < .001  [supports predicted heuristic/exemption]

### matrix_rescue

- BabyLlama: estimate = -2.485, t(31) = -25.26, p = < .001  [supports predicted heuristic/exemption]
- Qwen-2.5-7B: estimate = -3.627, t(31) = -9.33, p = < .001  [supports predicted heuristic/exemption]
- Llama-3.1-8B: estimate = -2.069, t(31) = -8.34, p = < .001  [supports predicted heuristic/exemption]
- Qwen-2.5-72B: estimate = -2.979, t(31) = -10.21, p = < .001  [supports predicted heuristic/exemption]
- Llama-3.1-70B: estimate = -1.448, t(31) = -5.02, p = < .001  [supports predicted heuristic/exemption]

### head_mismatch_penalty

- BabyLlama: estimate = 3.568, t(31) = 27.72, p = < .001  [supports predicted heuristic/exemption]
- Qwen-2.5-7B: estimate = 3.867, t(31) = 8.12, p = < .001  [supports predicted heuristic/exemption]
- Llama-3.1-8B: estimate = 3.462, t(31) = 11.43, p = < .001  [supports predicted heuristic/exemption]
- Qwen-2.5-72B: estimate = 3.342, t(31) = 9.45, p = < .001  [supports predicted heuristic/exemption]
- Llama-3.1-70B: estimate = 3.872, t(31) = 13.97, p = < .001  [supports predicted heuristic/exemption]

### proximity_trap

- BabyLlama: estimate = -1.503, t(31) = -14.47, p = < .001  [supports predicted heuristic/exemption]
- Qwen-2.5-7B: estimate = -2.639, t(31) = -7.79, p = < .001  [supports predicted heuristic/exemption]
- Llama-3.1-8B: estimate = -3.634, t(31) = -10.49, p = < .001  [supports predicted heuristic/exemption]
- Qwen-2.5-72B: estimate = -2.760, t(31) = -7.31, p = < .001  [supports predicted heuristic/exemption]
- Llama-3.1-70B: estimate = -2.936, t(31) = -10.20, p = < .001  [supports predicted heuristic/exemption]

### mismatch_penalty_coarg

- BabyLlama: estimate = 1.923, t(31) = 17.95, p = < .001  [supports predicted heuristic/exemption]
- Qwen-2.5-7B: estimate = 2.550, t(31) = 8.58, p = < .001  [supports predicted heuristic/exemption]
- Llama-3.1-8B: estimate = 3.242, t(31) = 10.96, p = < .001  [supports predicted heuristic/exemption]
- Qwen-2.5-72B: estimate = 3.618, t(31) = 9.05, p = < .001  [supports predicted heuristic/exemption]
- Llama-3.1-70B: estimate = 3.681, t(31) = 13.82, p = < .001  [supports predicted heuristic/exemption]

### mismatch_penalty_picture

- BabyLlama: estimate = 1.915, t(31) = 13.86, p = < .001  [supports predicted heuristic/exemption]
- Qwen-2.5-7B: estimate = 2.422, t(31) = 10.08, p = < .001  [supports predicted heuristic/exemption]
- Llama-3.1-8B: estimate = 1.990, t(31) = 11.67, p = < .001  [supports predicted heuristic/exemption]
- Qwen-2.5-72B: estimate = 1.959, t(31) = 7.08, p = < .001  [supports predicted heuristic/exemption]
- Llama-3.1-70B: estimate = 2.147, t(31) = 13.62, p = < .001  [supports predicted heuristic/exemption]

### logophoric_exemption

- BabyLlama: estimate = 0.008, t(31) = 0.11, p = 0.914
- Qwen-2.5-7B: estimate = 0.127, t(31) = 0.35, p = 0.725
- Llama-3.1-8B: estimate = 1.251, t(31) = 3.48, p = 0.002  [supports predicted heuristic/exemption]
- Qwen-2.5-72B: estimate = 1.659, t(31) = 3.31, p = 0.002  [supports predicted heuristic/exemption]
- Llama-3.1-70B: estimate = 1.534, t(31) = 5.50, p = < .001  [supports predicted heuristic/exemption]
