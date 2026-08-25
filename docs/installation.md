# 本地 / GPU 环境安装清单（验证 quantum_bench）

> 目的：把 `quantum_bench.py`（Phase 1：Ising vs pyMatching）跑通所需的环境装齐。
> 推荐在 **Linux + NVIDIA GPU** 机器上做；Windows 本地若只做语法/帮助检查，
> 至少装 `stim` + `pymatching` 才能真跑。

按顺序执行，每步确认通过再进下一步。

---

## 0. 前置检查
```bash
nvidia-smi                 # 能看到 GPU 型号 + CUDA 版本是对的
python3 --version          # >= 3.9
```

## 1. 虚拟环境
```bash
cd <repo>                  # 你 clone 的 Ising-Decoding 根目录
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS; Windows: .venv\Scripts\activate
```

## 2. 安装 CUDA 匹配版 PyTorch（关键，别用默认 pip）
先看你的 CUDA 版本（`nvidia-smi` 顶部 CUDA Version），按版本选 index-url：
```bash
# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8
# pip install torch --index-url https://download.pytorch.org/whl/cu118
# 验证 GPU：python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3. 安装项目依赖
```bash
# 方式 A：用 pyproject（推荐）
pip install -e ".[plots]"          # 核心 + 绘图
# 方式 B：手动装核心包
pip install numpy "torch>=2.0" huggingface-hub stim pymatching
```
> `stim`/`pymatching` 是纯 Python/C++ 包，无 GPU 也能装；真跑解码需要 GPU 加速（torch）。

## 4. 克隆 NVIDIA/ising-decoding（color_code 电路必备）
```bash
git clone https://github.com/NVIDIA/ising-decoding.git
# 路径假设：~/ising-decoding 或 D:\my_projects\ising-decoding
# 把它指向环境变量（quantum_bench 的 --code color_code 用）：
export NV_ISING_CODECODE_DIR="$(pwd)/ising-decoding/code"
# 或者运行时不设环境变量，直接传参：--nvidia-code-dir <repo>/code
```
> 只需其 `code/qec/color_code/...`（`build_color_memory_circuit`）。仅当你要跑 `--code color_code` 才需要。

## 5. 验证（surface_code 全链路，无需 NVIDIA 仓库）
```bash
python scripts/quantum_bench.py --self-test                 # 小规模 sanity check
python scripts/quantum_bench.py --decoder pymatching --distance 5 --shots 2000 --save
```
- `--self-test` 通过 = `stim` + `pymatching` 全链路 OK，输出**真实** throughput / logical error rate。

## 6. 验证 color_code（需要 NVIDIA 仓库 + 依赖）
```bash
python scripts/quantum_bench.py --decoder pymatching --code color_code \
    --distance 3 --shots 500 --basis X --save
```
- 若报 "NVIDIA color-code module not found" → 检查 `--nvidia-code-dir` / `NV_ISING_CODECODE_DIR` 指向 `<repo>/code`；
- 若报 "Failed to import ... builder" → 确认 `stim` 已装、NVIDIA `code/` 下 `qec` 可导入。

---

## 常见坑
| 现象 | 处理 |
|---|---|
| `torch.cuda.is_available()` 为 False | torch 装成了 CPU 版 → 用 `--index-url .../cuXXX` 重装；或驱动/CUDA 版本不匹配 |
| `ModuleNotFoundError: stim` | `pip install stim` |
| color_code 报找不到 `qec` | `--nvidia-code-dir` 指到 `<isorng-decoding>/code` 那一层，不是仓库根 |
| Ising 解码报 NotImplementedError | 正常——NVIDIA 3D-CNN 预处理/后处理尚未接入，见 `quantum_bench.py` 的插件点；不会给假 LER |
| 结果 CI 太宽 | 增加 `--shots`（≥2000） |

---

## 一张表速查（本机 vs 云）
| 依赖 | 本机(Windows, 无 GPU) | 云(租 A100/4090) |
|---|---|---|
| torch(CUDA) | CPU 版够跑 help/语法 | 必须 CUDA 版 |
| stim/pymatching | 装 | 装 |
| huggingface_hub | 装 | 装 |
| NVIDIA/ising-decoding | 只需 `--code color_code` 时 | 需要 |
```