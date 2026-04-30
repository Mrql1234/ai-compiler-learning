# AGENTS.md

æœ¬æ–‡ä»¶ç”¨äºè¯´æ˜å½“å‰ä»“åº“çš„å¼€å‘èƒŒæ™¯ã€ç›®å½•ç”¨é€”ã€å·²è½åœ°çº¦å®šä¸åç»­å¼€å‘å»ºè®®ã€‚ä½œç”¨èŒƒå›´ä¸º**ä»“åº“æ ¹ç›®å½•åŠå…¶æ‰€æœ‰å­ç›®å½•**ï¼›è‹¥æ›´æ·±å±‚ç›®å½•å­˜åœ¨æ–°çš„ `AGENTS.md`ï¼Œåˆ™ä»¥æ›´æ·±å±‚æ–‡ä»¶ä¸ºå‡†ã€‚

---

## 1. ä»“åº“æ¦‚è§ˆ

å½“å‰ä»“åº“æ˜¯ä¸€ä¸ªä»¥ **AI ç¼–è¯‘å™¨ / MLIR / Triton / LLVM å­¦ä¹ ä¸å®éªŒ** ä¸ºä¸»é¢˜çš„å·¥ä½œåŒºï¼Œä¸»è¦åŒ…å«ä»¥ä¸‹å†…å®¹ï¼š

- `projects/mlir-passes/`
  - MLIR å¤–éƒ¨ pass å­¦ä¹ é¡¹ç›®
  - é‡ç‚¹æ˜¯ C++ / MLIR pass å¼€å‘ã€IR å˜æ¢ã€`mlir-opt` éªŒè¯
- `projects/mini-ai-compiler/`
  - Python å®ç°çš„ç«¯åˆ°ç«¯æ•™å­¦å‹ AI ç¼–è¯‘å™¨é¡¹ç›®
  - å½“å‰æ˜¯æœ¬ä»“åº“é‡Œæœ€å®Œæ•´çš„â€œå‰ç«¯ -> IR -> pass -> backend -> éªŒè¯â€é—­ç¯åŸå‹
- `other/`
  - æ‚é¡¹èµ„æ–™ç›®å½•
  - **è¯¥ç›®å½•ä¸‹å·²æœ‰å•ç‹¬çš„ `other/AGENTS.md`ï¼Œè¿›å…¥è¯¥ç›®å½•æ ‘åéœ€è¦ä¼˜å…ˆéµå®ˆå®ƒ**

æ ¹ç›®å½•ä¸‹è¿˜æœ‰ä¸€äº›ç¯å¢ƒè®°å½•æ–‡ä»¶ï¼Œä¾‹å¦‚ï¼š

- `README.md`
- `WSL_MLIR_SETUP_LOG.md`

---

## 2. å½“å‰é‡ç‚¹é¡¹ç›®ï¼š`projects/mini-ai-compiler/`

### 2.1 é¡¹ç›®å®šä½

`mini-ai-compiler` æ˜¯ä¸€ä¸ª**åŒè½¨æ¶æ„**é¡¹ç›®ï¼Œç›®æ ‡æ˜¯å®ç°ä¸€ä¸ªé¢å‘å°æ¨¡å‹å­é›†çš„ç«¯åˆ°ç«¯ AI ç¼–è¯‘å™¨ã€‚

é¡¹ç›®é¢˜ç›®å¯ä»¥æ¦‚æ‹¬ä¸ºï¼š

> Mini AI Compiler: From ONNX / PyTorch FX to MLIR IR, Optimized CPU/Triton Execution

å½“å‰åˆ†ä¸ºä¸¤æ¡è½¨é“ï¼š

- Python è½¨ï¼šæ•™å­¦åŸå‹ã€bridgeã€reference backendã€éªŒè¯ä¸ benchmark
- MLIR è½¨ï¼šæ­£å¼ç¼–è¯‘å™¨ä¸»çº¿ï¼Œä½äº `projects/mini-ai-compiler/compiler-mlir/`

### 2.2 å½“å‰èŒƒå›´

å½“å‰é¡¹ç›®**ä¸æ˜¯å®Œæ•´å¤§æ¨¡å‹ç¼–è¯‘å™¨**ï¼Œè€Œæ˜¯é¢å‘ä¸€ä¸ªå¯æ§çš„å°å­é›†ï¼š

- æ¨èæ¨¡å‹èŒƒå›´ï¼š
  - `MLP / FFN`
  - `single attention block`
- å½“å‰çœŸæ­£è·‘é€šçš„ç¤ºä¾‹ï¼š
  - `TinyMLP`

### 2.3 å½“å‰å®ç°çŠ¶æ€

å½“å‰é¡¹ç›®å·²ç»å…·å¤‡ä»¥ä¸‹èƒ½åŠ›ï¼š

- `PyTorch FX` å‰ç«¯å¯¼å…¥
- è‡ªå®šä¹‰ IR
- `Constant Folding`
- `DCE`
- `linear + relu` çš„ `FusionPass`
- CPU reference backend
- IR æ–‡æœ¬ dump
- benchmark è„šæœ¬
- ONNX importer MVP ä»£ç æ¡†æ¶ï¼ˆè¿è¡Œä¾èµ– `onnx` åŒ…ï¼‰
- MLIR é£æ ¼æ–‡æœ¬å¯¼å‡º
- MLIR é£æ ¼ rewrite åŸå‹
- `compiler-mlir/` out-of-tree MLIR å­å·¥ç¨‹éª¨æ¶

å½“å‰è¿˜**æ²¡æœ‰**çœŸæ­£è¿›å…¥ï¼š

- æ­£å¼ MLIR pass å®è£…
- `MLIR -> LLVM` CPU é—­ç¯
- `MLIR -> Triton/GPU` æ­£å¼é—­ç¯

---

## 3. `mini-ai-compiler` ç›®å½•ç»“æ„è¯´æ˜

å½“å‰ `projects/mini-ai-compiler/` ä¸»è¦ç»“æ„å¦‚ä¸‹ï¼š

- `requirements.md`
  - éœ€æ±‚è§„æ ¼
- `design.md`
  - æ¶æ„ä¸è®¾è®¡è¯´æ˜
- `tasks.md`
  - åˆ†é˜¶æ®µä»»åŠ¡æ¸…å•
- `README.md`
  - é¡¹ç›®è¯´æ˜ä¸ä½¿ç”¨æ–¹å¼
- `frontend/`
  - å‰ç«¯å¯¼å…¥å™¨
  - å½“å‰åŒ…æ‹¬ï¼š
    - `fx_importer.py`
    - `onnx_importer.py`
- `ir/`
  - è‡ªå®šä¹‰ IR å®šä¹‰
  - åŒ…æ‹¬ï¼š
    - `graph.py`
    - `node.py`
    - `value.py`
    - `types.py`
    - `printer.py`
- `passes/`
  - å›¾ä¼˜åŒ– pass
  - å½“å‰åŒ…æ‹¬ï¼š
    - `constant_fold.py`
    - `dce.py`
    - `fusion.py`
    - `manager.py`
- `backend/cpu/`
  - CPU reference backend
- `compiler-mlir/`
  - æ­£å¼ MLIR C++ å­å·¥ç¨‹
  - ä½¿ç”¨ out-of-tree LLVM/MLIR CMake ä½“ç³»
- `examples/`
  - ç¤ºä¾‹æ¨¡å‹
  - å½“å‰ä¸»è¦æ˜¯ `mlp.py`
- `tools/`
  - è¿è¡Œä¸è°ƒè¯•å·¥å…·
  - å½“å‰åŒ…æ‹¬ï¼š
    - `run_mlp_example.py`
    - `dump_ir.py`
- `benchmarks/`
  - benchmark è„šæœ¬
  - å½“å‰ä¸»è¦æ˜¯ `bench_mlp.py`
- `tests/`
  - åŸºç¡€å•æµ‹

---

## 4. `mini-ai-compiler` çš„æ ¸å¿ƒè®¾è®¡çº¦å®š

### 4.1 IR è®¾è®¡çº¦å®š

å½“å‰é¡¹ç›®é‡‡ç”¨åŒå±‚ IR è§†è§’ï¼š

- Python åŸå‹ IRï¼š
  - é‡‡ç”¨â€œ**Node è¡¨ç¤ºæ“ä½œï¼ŒValue è¡¨ç¤ºæ•°æ®è¾¹**â€çš„æ€è·¯
- MLIR ä¸»çº¿ IRï¼š
  - é‡‡ç”¨ `mini` dialect ä¸ MLIR module/pass/lowering ä½“ç³»

Python åŸå‹ IR å…·ä½“çº¦å®šå¦‚ä¸‹ï¼š

- `Node`
  - è¡¨ç¤ºä¸€ä¸ªæ“ä½œ / ç®—å­
  - ä¾‹å¦‚ï¼š`constant`ã€`add`ã€`mul`ã€`linear`ã€`relu`
- `Value`
  - è¡¨ç¤ºæŸä¸ªæ“ä½œäº§ç”Ÿçš„ç»“æœ
  - æˆ–å›¾è¾“å…¥å€¼
- `Graph`
  - ç»´æŠ¤ï¼š
    - `inputs`
    - `outputs`
    - `nodes`

è¿™æ˜¯ä¸€ä¸ªå…¸å‹çš„å›¾ IRï¼š

- ç‚¹ = `Node`
- è¾¹ = `Value`

### 4.2 å½“å‰ CPU backend çš„è¯­ä¹‰çº¦å®š

å½“å‰ CPU backend åŸºäº `numpy` åš reference æ‰§è¡Œã€‚

éœ€è¦æ³¨æ„ï¼š

- `matmul`
  - ä½¿ç”¨ `args[0] @ args[1]`
- `linear`
  - ä½¿ç”¨ `args[0] @ args[1].T`
  - åŸå› æ˜¯ PyTorch `Linear.weight` å­˜å‚¨ä¸º `[out_features, in_features]`
- `fused_linear_relu`
  - å…ˆåšçº¿æ€§å±‚ï¼Œå†åš `relu`

### 4.3 å½“å‰ fusion çº¦å®š

å½“å‰åªå®ç°äº†ä¸€ä¸ªæœ€å° fusion è§„åˆ™ï¼š

- `linear + relu`
  -> `fused_linear_relu`

è¿™æ˜¯ä¸€ä¸ª**æ•™å­¦å‹èåˆç¤ºä¾‹**ï¼Œç›®çš„ä¸»è¦æ˜¯ï¼š

- å±•ç¤º pass å¦‚ä½•æ”¹å›¾
- å±•ç¤ºä¼˜åŒ–å‰å IR å·®å¼‚
- ç»™åç»­ Triton / fused op è·¯çº¿æ‰“åŸºç¡€

MLIR è½¨å½“å‰åˆ™å·²æ–°å¢ï¼š

- dialect skeleton
- pass registration skeleton
- compiler driver skeleton
- smoke test skeleton

### 4.4 å½“å‰ DCE æ–¹æ³•

å½“å‰ DCE ä½¿ç”¨çš„æ˜¯ï¼š

- **ä»å›¾è¾“å‡ºåå‘åšå¯è¾¾æ€§ / æ´»è·ƒæ€§ä¼ æ’­**

å³ï¼š

- ä» `graph.outputs` å‡ºå‘
- æ²¿ç€ `Value -> producer Node -> input Values`
- åå‘è¿½è¸ªæ‰€æœ‰æ´»èŠ‚ç‚¹
- æœªè¢«è¿½è¸ªåˆ°çš„èŠ‚ç‚¹åˆ é™¤

è¿™æ˜¯é’ˆå¯¹å½“å‰â€œæ— å¤æ‚æ§åˆ¶æµã€ä»¥æ•°æ®æµä¸ºä¸»â€çš„ç¥ç»ç½‘ç»œå›¾éå¸¸åˆé€‚çš„æ–¹æ³•ã€‚

---

## 5. å½“å‰é˜¶æ®µçŠ¶æ€

### Phase 1

å½“å‰ Phase 1 å·²å®Œæˆï¼š

- é¡¹ç›®éª¨æ¶
- è‡ªå®šä¹‰ IR
- FX importer MVP
- CPU backend MVP
- `constant_fold`
- `dce`
- MLP ç¤ºä¾‹
- åŸºç¡€æµ‹è¯•

### Phase 2

å½“å‰ Phase 2 å·²å®Œæˆç¬¬ä¸€æ‰¹ï¼š

- `FusionPass`
- IR dump å¢å¼º
- benchmark è„šæœ¬
- ONNX importer MVP ä»£ç 

### Phase 3 / 4

å½“å‰ Phase 3 / 4 å·²å®Œæˆâ€œæ¶æ„å‡çº§çš„ç¬¬ä¸€æ‰¹â€ï¼š

- MLIR é£æ ¼æ–‡æœ¬å¯¼å‡º
- MLIR é£æ ¼ rewrite åŸå‹
- Triton lowering / executor éª¨æ¶
- `compiler-mlir/` å­å·¥ç¨‹éª¨æ¶

ä½†è¿˜æ²¡æœ‰å®Œæˆæ­£å¼ä¸»é“¾è·¯ï¼š

- MLIR-native pass å®è£…
- `MLIR -> LLVM` CPU è·¯çº¿
- `MLIR -> Triton/GPU` æ­£å¼è·¯çº¿

### Phase 3

å½“å‰å°šæœªå¼€å§‹çœŸæ­£å®ç°ï¼š

- Triton kernel MVP
- Triton executor
- fused op lowering
- CPU / Triton å¯¹ç…§ benchmark

### Phase 4

å½“å‰å°šæœªå¼€å§‹çœŸæ­£å®ç°ï¼š

- MLIR é£æ ¼ IR è¾“å‡º
- IR åˆ° MLIR æ¦‚å¿µæ˜ å°„
- MLIR-based pass è¿ç§»

---

## 6. å¼€å‘å»ºè®®

### 6.1 ä¿®æ”¹ `mini-ai-compiler` æ—¶çš„åŸåˆ™

- ä¼˜å…ˆä¿æŒâ€œ**æ•™å­¦æ€§ + å¯è¯»æ€§ + æœ€å°é—­ç¯**â€
- å…ˆä¿®ä¸»é“¾è·¯ï¼Œä¸è¦è¿‡æ—©å¼•å…¥å¤æ‚æŠ½è±¡
- ä¿æŒä»£ç å°è€Œæ¸…æ™°ï¼Œé¿å…ä¸ºäº†â€œåƒå·¥ä¸šæ¡†æ¶â€è€Œè¿‡åº¦è®¾è®¡
- ä»»ä½•æ–°å¢èƒ½åŠ›æœ€å¥½é…ï¼š
  - ç¤ºä¾‹
  - æµ‹è¯•
  - æˆ– dump/benchmark å…¥å£

### 6.2 åç»­æ¨èé¡ºåº

å¦‚æœç»§ç»­æ¨è¿› `mini-ai-compiler`ï¼Œå»ºè®®é¡ºåºæ˜¯ï¼š

1. ç¨³å®š Python bridge è¾“å‡º
2. åœ¨ `compiler-mlir/` ä¸­å®ç°æ­£å¼ dialect / pass
3. æ‰“é€š `MLIR -> LLVM` CPU é—­ç¯
4. å†æ‰“é€š Triton/GPU è·¯çº¿
5. æœ€åç»Ÿä¸€ Python harness éªŒè¯ä¸ benchmark

### 6.3 ä¸å»ºè®®çš„æ–¹å‘

å½“å‰é˜¶æ®µä¸å»ºè®®ï¼š

- ç›´æ¥å¤§æ”¹æˆå¤æ‚æ¡†æ¶å¼æ¶æ„
- ä¸€å¼€å§‹å°±è¿½æ±‚å®Œæ•´ Transformer / å®Œæ•´ Qwen æ”¯æŒ
- ç›´æ¥åˆ‡åˆ°å®Œæ•´ MLIR-native å®ç°è€Œæ”¾å¼ƒç°æœ‰ Python åŸå‹

---

## 7. è¿è¡Œä¸éªŒè¯ä¹ æƒ¯

åœ¨ `projects/mini-ai-compiler/` ä¸‹ï¼Œå¸¸ç”¨å‘½ä»¤åŒ…æ‹¬ï¼š

- è¿è¡Œ MLP ç¤ºä¾‹ï¼š
  - `python3 -m tools.run_mlp_example`
- è¾“å‡º IRï¼š
  - `python3 -m tools.dump_ir`
- è·‘ benchmarkï¼š
  - `python3 -m benchmarks.bench_mlp`
- è·‘æµ‹è¯•ï¼š
  - `python3 -m unittest discover -s tests`

ä¾èµ–æ–¹é¢ï¼š

- å½“å‰è‡³å°‘éœ€è¦ï¼š
  - `numpy`
  - `torch`
- è‹¥è¦éªŒè¯ ONNX importerï¼Œè¿˜éœ€è¦ï¼š
  - `onnx`

---

## 8. ç»™åç»­åä½œè€…çš„æé†’

- å¦‚æœåªæ˜¯æƒ³å­¦ä¹ â€œæœ€å°é—­ç¯åŸå‹â€ï¼Œä¼˜å…ˆçœ‹ `mini-ai-compiler` çš„ Python è½¨
- å¦‚æœæƒ³å­¦ä¹ â€œæ›´çœŸå®çš„ç¼–è¯‘å™¨ä¸»çº¿â€ï¼Œä¼˜å…ˆçœ‹ `mini-ai-compiler/compiler-mlir/`
- å¦‚æœæ˜¯ç ”ç©¶ MLIR C++ passï¼Œè¯·çœ‹ `projects/mlir-passes/`
- å¦‚æœæ˜¯ç ”ç©¶ Triton kernelï¼Œå¯æŠŠ `mini-ai-compiler` è§†ä½œå‰ç«¯ / IR / pass åŸå‹ï¼Œå†ä¸ Triton å®éªŒç»“åˆ

åç»­ä¿®æ”¹æ—¶ï¼Œå»ºè®®ä¼˜å…ˆåŒæ­¥æ›´æ–°ï¼š

- `requirements.md`
- `design.md`
- `tasks.md`
- `README.md`

é¿å…ä»£ç å’Œæ–‡æ¡£é•¿æœŸè„±èŠ‚ã€‚

---

## 9. Cursor / clangd / WSL ÅäÖÃÔ¼¶¨

ÕâÒ»½ÚÊÇÕë¶Ô `projects/mini-ai-compiler/compiler-mlir/` ºÍ `projects/mlir-passes/` µÄ C++/MLIR ¿ª·¢»·¾³Ô¼¶¨£¬Ä¿µÄÊÇ±ÜÃâ¡°ÀàÌø×ªµ½´íÎóÍ·ÎÄ¼ş¡±¡°clangd ¶Á´í compile_commands.json¡±¡°Ã÷Ã÷ÔÚ WSL ÀïÈ´ÓÃÁË apt °æË÷Òı¡±ÕâÀàÎÊÌâ¡£

### 9.1 ±ØĞëÊ¹ÓÃ WSL Remote£¬¶ø²»ÊÇ `\\wsl$` ±¾µØÂ·¾¶Ä£Ê½

- ÕıÈ··½Ê½£º
  - ÔÚ Cursor ×óÏÂ½Ç¿´µ½ `WSL: Ubuntu-22.04`
  - ÒÔ Remote-WSL ·½Ê½´ò¿ª `/home/ql/code/ai-compiler-learning`
- ²»ÍÆ¼ö·½Ê½£º
  - Ö±½ÓÓÃ Windows ±¾µØ´°¿Ú´ò¿ª `\\wsl$\\Ubuntu-22.04\\home\\ql\\code\\ai-compiler-learning`

Ô­Òò£º

- ±¾²Ö¿âÀïµÄ clangd / CMake / LLVM / MLIR Í·ÎÄ¼şÂ·¾¶¶¼ÊÇ Linux Â·¾¶
- Ö»ÓĞÕæÕıµÄ WSL Remote Ä£Ê½£¬clangd ²Å»áÎÈ¶¨°´ Linux »·¾³½âÎö

### 9.2 ²»ÒªÔÚÕû¸ö¹¤×÷ÇøÀïĞ´ËÀÈ«¾Ö `--compile-commands-dir`

Ôø¾­²È¹ıµÄ¿Ó£º

- `.vscode/settings.json` ÀïÈç¹ûĞ´ËÀ£º
  - `--compile-commands-dir=${workspaceFolder}/projects/mlir-passes/build-wsl`
- ÄÇÃ´Õû¸ö²Ö¿âËùÓĞ C++ ÎÄ¼ş¶¼»á±» clangd Ç¿ÖÆ°´ `mlir-passes` µÄ±àÒëÊı¾İ¿âÍÆ¶Ï
- ½á¹û»áµ¼ÖÂ£º
  - `compiler-mlir` µÄÎÄ¼ş±»´íÎóµØ¡°ÍÆ¶Ï×Ô¡± `projects/mlir-passes/...`
  - Ìø×ªºÍÕï¶ÏÅÜµ½ apt °æ LLVM/MLIR Í·ÎÄ¼ş

µ±Ç°Ô¼¶¨£º

- `clangd.arguments` Àï²»ÒªÔÙÉèÖÃÈ«¾Ö `--compile-commands-dir`
- ÈÃÃ¿¸ö×ÓÏîÄ¿Í¨¹ı×Ô¼ºµÄ `.clangd` + `compile_commands.json` ¾ö¶¨±àÒëÊı¾İ¿â

### 9.3 `compiler-mlir` Ó¦Ê¹ÓÃÄÄÌ× LLVM/MLIR

`projects/mini-ai-compiler/compiler-mlir/` µ±Ç°Ó¦°ó¶¨µ½Ô´Âë°æ LLVM/MLIR£¬¶ø²»ÊÇ apt °æ¡£

ÕıÈ·À´Ô´£º

- LLVM Ô´Âë£º
  - `/home/ql/code/llvm_clang_static_analyzer/llvm`
- MLIR Ô´Âë£º
  - `/home/ql/code/llvm_clang_static_analyzer/mlir`
- MLIR build£º
  - `/home/ql/code/llvm_clang_static_analyzer/build-mlir`

ÕıÈ·µÄ `compiler-mlir` CMake ÅäÖÃ£º

- `LLVM_DIR=/home/ql/code/llvm_clang_static_analyzer/build-mlir/lib/cmake/llvm`
- `MLIR_DIR=/home/ql/code/llvm_clang_static_analyzer/build-mlir/lib/cmake/mlir`

ÖØĞÂÅäÖÃÃüÁî£º

- `cd ~/code/ai-compiler-learning/projects/mini-ai-compiler/compiler-mlir`
- `cmake -S . -B build -G Ninja -DLLVM_DIR=/home/ql/code/llvm_clang_static_analyzer/build-mlir/lib/cmake/llvm -DMLIR_DIR=/home/ql/code/llvm_clang_static_analyzer/build-mlir/lib/cmake/mlir`
- `cmake --build build -j2`

### 9.4 `compiler-mlir` µ±Ç° clangd ÅäÖÃÀ´Ô´

`projects/mini-ai-compiler/compiler-mlir/` µ±Ç°Ö÷ÒªÒÀÀµÈı´¦£º

- ¸ùÄ¿Â¼ `.clangd`
  - Õë¶Ô `projects/mini-ai-compiler/compiler-mlir/.*` Ìí¼ÓÔ´Âë°æ LLVM/MLIR include
- `projects/mini-ai-compiler/compiler-mlir/.clangd`
  - Ö¸¶¨ `CompilationDatabase: build`
- `projects/mini-ai-compiler/compiler-mlir/build/compile_commands.json`
  - ÕæÕı¾ö¶¨Ã¿¸ö `.cpp` ÎÄ¼şµÄ±àÒëÃüÁî

´ËÍâ»¹ÓĞ£º

- `projects/mini-ai-compiler/compiler-mlir/compile_flags.txt`
  - ×÷Îª fallback flags
  - Ò²±ØĞëÖ¸Ïò `llvm_clang_static_analyzer`£¬²»ÄÜ²ĞÁô `/usr/lib/llvm-15/include`

### 9.5 ÈçºÎÅĞ¶Ï clangd µ±Ç°ÊÇ·ñÕæµÄ×ß¶ÔÁË

¿´ clangd ÈÕÖ¾Ê±£¬ÖØµã¼ì²é£º

- ²»ÄÜÔÙ³öÏÖ£º
  - `--compile-commands-dir=.../projects/mlir-passes/build-wsl`
- ´ò¿ª `compiler-mlir` ÎÄ¼şÊ±£¬Ó¦¿´µ½£º
  - `Loaded compilation database from /home/ql/code/ai-compiler-learning/projects/mini-ai-compiler/compiler-mlir/build/compile_commands.json`
- ´¦Àí `MiniDialect.cpp` µÈÎÄ¼şÊ±£¬±àÒëÃüÁîÓ¦°üº¬£º
  - `/home/ql/code/llvm_clang_static_analyzer/llvm/include`
  - `/home/ql/code/llvm_clang_static_analyzer/mlir/include`
  - `/home/ql/code/llvm_clang_static_analyzer/build-mlir/include`

Èç¹ûÈÕÖ¾Àï³öÏÖ£º

- `building file ... MiniDialect.cpp with command inferred from ... projects/mlir-passes/...`

ËµÃ÷ clangd ÈÔÔÚ´íÎó¸´ÓÃ `mlir-passes` µÄ±àÒëÊı¾İ¿â¡£

### 9.6 Í·ÎÄ¼şÖ±½Ó°üº¬Ô­Ôò

¶ÔÓÚ `compiler-mlir` µÄ C++ ÎÄ¼ş£º

- ²»ÒªÒÀÀµ¡°´«µİ include Ç¡ºÃ¿ÉÓÃ¡±
- Ä³¸ö·ûºÅÔÚµ±Ç° `.cpp` ÀïÖ±½ÓÊ¹ÓÃÁË£¬¾ÍÓ¦ÏÔÊ½°üº¬¶ÔÓ¦Í·ÎÄ¼ş

Ô­Òò£º

- ÕæÕı±àÒëÓĞÊ±ÒòÎª´«µİ°üº¬¡°½ÄĞÒÍ¨¹ı¡±
- µ« clangd µÄÓïÒå·ÖÎö¸üÈİÒ×ÒòÎªÈ±ÉÙÖ±½ÓÍ·ÎÄ¼ş¶ø³öÏÖ¼Ù±¨´í¡¢Ìø×ªÒì³£

ÀıÈç±¾ÏîÄ¿ÖĞ£º

- `MiniDialect.cpp` Ê¹ÓÃ attribute Ïà¹ØÀàĞÍÊ±£¬ĞèÒªÏÔÊ½°üº¬£º
  - `mlir/IR/BuiltinAttributes.h`

### 9.7 Èç¹û clangd / Ìø×ªÔÙ´ÎÒì³££¬½¨ÒéÅÅ²éË³Ğò

1. È·ÈÏ Cursor ×óÏÂ½ÇÊÇ·ñÊÇ `WSL: Ubuntu-22.04`
2. ¼ì²é `.vscode/settings.json` ÊÇ·ñÓÖĞ´ÈëÁËÈ«¾Ö `--compile-commands-dir`
3. ¼ì²é `compiler-mlir/build/compile_commands.json` ÊÇ·ñÈÔÖ¸ÏòÔ´Âë°æ LLVM/MLIR
4. ¼ì²é¸ùÄ¿Â¼ `.clangd` ÊÇ·ñÎó¼ÓÁË `/usr/lib/llvm-15/include`
5. ¼ì²é `projects/mini-ai-compiler/compiler-mlir/compile_flags.txt`
6. ÖØÆô clangd£º
   - `clangd: Restart language server`
7. ÈôÈÔÒì³££¬ÔÙÇå»º´æ£º
   - WSL: `/home/ql/.cache/clangd`
   - Cursor workspaceStorage ÖĞ¶ÔÓ¦µ±Ç°¹¤×÷ÇøµÄÄ¿Â¼

### 9.8 µ±Ç°ÒÑÖª½áÂÛ

- `projects/mlir-passes/` ¿ÉÒÔ¼ÌĞøÊ¹ÓÃ apt °æ / ÏÖÓĞ `build-wsl`
- `projects/mini-ai-compiler/compiler-mlir/` Ó¦ÓÅÏÈÊ¹ÓÃ `llvm_clang_static_analyzer/build-mlir`
- Í¬Ò»²Ö¿âÏÂÓĞ¶à¸ö C++ ×ÓÏîÄ¿Ê±£¬clangd ±ØĞë°´¡°×ÓÏîÄ¿¸÷×ÔµÄÊı¾İ¿â¡±¹¤×÷£¬²»ÄÜÓÃÒ»¸öÈ«¾Ö `compile-commands-dir` Ç¿ĞĞ¸²¸Ç
