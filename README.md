DeepGEMM 项目的中文翻译：

# DeepGEMM

DeepGEMM 是一个专为干净且高效的 FP8 通用矩阵乘法（GEMMs）设计的库，具有细粒度缩放功能，如 [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) 中所提出的。它支持普通和混合专家（MoE）分组 GEMMs。该库用 CUDA 编写，安装过程中无需编译，通过轻量级的即时（JIT）模块在运行时编译所有内核。

目前，DeepGEMM 仅支持 NVIDIA Hopper 张量核心。为了解决不精确的 FP8 张量核心累积问题，它采用了 CUDA 核心两级累积（提升）。虽然它利用了 [CUTLASS](https://github.com/nvidia/cutlass) 和 [CuTe](https://github.com/NVIDIA/cutlass/tree/main/include/cute) 的一些概念，但它避免了对它们的模板或代数的严重依赖。相反，该库的设计注重简洁性，只有一个核心内核函数，包含大约 **~300 行代码**。这使得它成为学习 Hopper FP8 矩阵乘法和优化技术的干净且易于访问的资源。

尽管设计轻巧，DeepGEMM 的性能在各种矩阵形状上都与专家调整的库相当或更好。

## 性能

我们在 H800 上使用 NVCC 12.8 测试了 DeepSeek-V3/R1 推理中可能使用的所有形状（包括预填充和解码，但不包括张量并行）。所有加速指标均与我们基于 CUTLASS 3.6 的内部精心优化的实现进行比较计算。

DeepGEMM 在某些形状上表现不佳，如果您有兴趣，欢迎提交优化 PR。

### 用于密集模型的普通 GEMMs

|  M   |   N   |   K   | 计算量       | 内存带宽     | 加速比 |
|:----:|:-----:|:-----:|:-----------:|:----------------:|:-------:|
|  64  | 2112  | 7168  | 206 TFLOPS  |    1688 GB/s     |  2.7x   |
|  64  | 24576 | 1536  | 289 TFLOPS  |    2455 GB/s     |  1.7x   |
|  64  | 32768 |  512  | 219 TFLOPS  |    2143 GB/s     |  1.8x   |
|  64  | 7168  | 16384 | 336 TFLOPS  |    2668 GB/s     |  1.4x   |
|  64  | 4096  | 7168  | 287 TFLOPS  |    2320 GB/s     |  1.4x   |
|  64  | 7168  | 2048  | 295 TFLOPS  |    2470 GB/s     |  1.7x   |
| 128  | 2112  | 7168  | 352 TFLOPS  |    1509 GB/s     |  2.4x   |
| 128  | 24576 | 1536  | 535 TFLOPS  |    2448 GB/s     |  1.6x   |
| 128  | 32768 |  512  | 358 TFLOPS  |    2103 GB/s     |  1.5x   |
| 128  | 7168  | 16384 | 645 TFLOPS  |    2604 GB/s     |  1.4x   |
| 128  | 4096  | 7168  | 533 TFLOPS  |    2221 GB/s     |  2.0x   |
| 128  | 7168  | 2048  | 510 TFLOPS  |    2277 GB/s     |  1.7x   |
| 4096 | 2112  | 7168  | 1058 TFLOPS |     527 GB/s     |  1.1x   |
| 4096 | 24576 | 1536  | 990 TFLOPS  |     786 GB/s     |  1.0x   |
| 4096 | 32768 |  512  | 590 TFLOPS  |    1232 GB/s     |  1.0x   |
| 4096 | 7168  | 16384 | 1358 TFLOPS |     343 GB/s     |  1.2x   |
| 4096 | 4096  | 7168  | 1304 TFLOPS |     500 GB/s     |  1.1x   |
| 4096 | 7168  | 2048  | 1025 TFLOPS |     697 GB/s     |  1.1x   |

### 用于 MoE 模型的 Grouped GEMMs (连续布局)

| 组数 | 每组 M  |  N   |  K   | 计算量       | 内存带宽     | 加速比 |
|:----:|:------:|:----:|:----:|:-----------:|:----------------:|:-------:|
|  4  |  8192  | 4096 | 7168 | 1297 TFLOPS |     418 GB/s     |  1.2x   |
|  4  |  8192  | 7168 | 2048 | 1099 TFLOPS |     681 GB/s     |  1.2x   |
|  8  |  4096  | 4096 | 7168 | 1288 TFLOPS |     494 GB/s     |  1.2x   |
|  8  |  4096  | 7168 | 2048 | 1093 TFLOPS |     743 GB/s     |  1.1x   |

### 用于 MoE 模型的 Grouped GEMMs (掩码布局)

| 组数 | 每组 M  |  N   |  K   | 计算量       | 内存带宽     | 加速比 |
|:----:|:------:|:----:|:----:|:-----------:|:----------------:|:-------:|
|  1  |  1024  | 4096 | 7168 | 1233 TFLOPS |     924 GB/s     |  1.2x   |
|  1  |  1024  | 7168 | 2048 | 925 TFLOPS  |     968 GB/s     |  1.2x   |
|  2  |   512  | 4096 | 7168 | 1040 TFLOPS |    1288 GB/s     |  1.2x   |
|  2  |   512  | 7168 | 2048 | 916 TFLOPS  |    1405 GB/s     |  1.2x   |
|  4  |   256  | 4096 | 7168 | 932 TFLOPS  |    2064 GB/s     |  1.1x   |
|  4  |   256  | 7168 | 2048 | 815 TFLOPS  |    2047 GB/s     |  1.2x   |

## 快速开始

### 要求

- Hopper 架构 GPU，必须支持 `sm_90a`
- Python 3.8 或更高版本
- CUDA 12.3 或更高版本
  - **但强烈建议使用 12.8 或更高版本以获得最佳性能**
- PyTorch 2.1 或更高版本
- CUTLASS 3.6 或更高版本 (可以通过 Git 子模块克隆)

### 开发

```bash
# 必须克隆子模块
git clone --recursive git@github.com:deepseek-ai/DeepGEMM.git

# 为第三方 (CUTLASS 和 CuTe) 包含目录创建符号链接
python setup.py develop

# 测试 JIT 编译
python tests/test_jit.py

# 测试所有 GEMM 实现 (普通、连续分组和掩码分组)
python tests/test_core.py
```

### 安装

```bash
python setup.py install
```

然后，在您的 Python 项目中导入 `deep_gemm`，即可开始使用！

## 接口

#### 注意事项

该库仅包含 GEMM 内核。它要求左侧 (LHS) 缩放因子进行 TMA 对齐并转置，并且仅支持 NT 格式（非转置 LHS 和转置 RHS）。对于转置或其他 FP8 类型转换操作，请独立实现或将它们融合到之前的内核中。虽然该库提供了一些简单的 PyTorch 实用函数，但这些函数可能会导致性能下降，但我们的主要重点是优化 GEMM 内核本身。

#### 普通密集 GEMMs (非分组)

要执行基本的非分组 FP8 GEMM，请调用 `deep_gemm.gemm_fp8_fp8_bf16_nt` 函数。有关更多详细信息，请参阅函数文档。

#### Grouped GEMMs (连续布局)

与 CUTLASS 中的传统分组 GEMMs 不同，DeepGEMM 仅对 M 轴进行分组，而 N 和 K 必须保持固定。此设计适用于 MoE 模型中专家共享相同形状的场景。

对于训练前向传递或推理预填充，每个专家可能会处理不同数量的标记，我们将这些标记连接到一个张量中，称为“连续”布局。请注意，每个专家段必须与 GEMM M 块大小对齐（`get_m_alignment_for_contiguous_layout()`）。

有关更多信息，请参阅 `m_grouped_gemm_fp8_fp8_bf16_nt_contiguous` 函数文档。

#### Grouped GEMMs (掩码布局)

在推理的解码阶段，当启用 CUDA 图并且 CPU 不知道每个专家接收的标记数量时，我们支持掩码分组 GEMMs。通过提供掩码张量，内核仅计算有效部分。

为此，请使用 `m_grouped_gemm_fp8_fp8_bf16_nt_masked` 并查阅相关文档。一个示例用法是使用来自 [DeepEP](https://github.com/deepseek-ai/DeepEP) 的低延迟内核的输出作为输入。

#### 实用函数

除了上述内核之外，该库还提供了一些实用函数：

- `deep_gemm.set_num_sms`: 设置要使用的最大 SM 数量
- `deep_gemm.get_num_sms`: 获取当前 SM 最大数量
- `deep_gemm.get_m_alignment_for_contiguous_layout`: 获取分组连续布局的组级对齐要求
- `deep_gemm.get_tma_aligned_size`: 获取所需的 TMA 对齐大小
- `deep_gemm.get_col_major_tma_aligned_tensor`: 获取一个列主序的 TMA 对齐张量

该库还提供了一些环境变量，这些变量可能有用：

- `DG_CACHE_DIR`: 字符串，存储编译内核的缓存目录，默认为 `$HOME/.deep_gemm`
- `DG_NVCC_COMPILER`: 字符串，指定的 NVCC 编译器路径；默认情况下将在 `from torch.utils.cpp_extension.CUDA_HOME` 中查找
- `DG_DISABLE_FFMA_INTERLEAVE`: 0 或 1，禁用 FFMA 交错优化
- `DG_PTXAS_VERBOSE`: 0 或 1，显示详细的 PTXAS 编译器输出
- `DG_PRINT_REG_REUSE`: 0 或 1，打印 FFMA 交错详细信息
- `DG_JIT_PRINT_NVCC_COMMAND`: 0 或 1，打印 NVCC 编译命令
- `DG_JIT_DEBUG`: 0 或 1，打印更多调试信息

有关其他示例和详细信息，请参阅 [测试代码](tests/test_core.py) 或查看相应的 Python 文档。

## 优化

我们用 🐳 标记出 CUTLASS 中未包含的技术。

#### 持久 warp 专业化

遵循 CUTLASS 设计，DeepGEMM 中的内核是 warp 专业化的，支持重叠数据移动、张量核心 MMA 指令和 CUDA 核心提升。下面显示了一个说明此过程的简化图：

![design](figures/design.png)

#### Hopper TMA 特性

[张量内存加速器](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#tensor-memory-accelerator) (TMA) 是 Hopper 架构引入的一项新硬件功能，旨在实现更快、异步的数据移动。具体来说，我们利用 TMA 进行：

- TMA 加载 LHS、LHS 缩放因子和 RHS 矩阵
- TMA 存储输出矩阵
- TMA 多播（仅限于 LHS 矩阵）
- TMA 描述符预取

#### 通用细节优化

- 使用 [`stmatrix`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-store-instruction-stmatrix) PTX 指令
- 针对不同 warpgroup 定制的[寄存器数量控制](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg)
- 尽可能多地重叠，例如重叠 TMA 存储和非 TMA RHS 缩放因子加载 🐳

#### 统一且优化的块调度器

- [一个调度器](deep_gemm/include/deep_gemm/scheduler.cuh) 用于所有非分组和分组内核
- [光栅化](https://github.com/NVIDIA/cutlass/blob/eefa171318b79cbe2e78514d4cce5cd0fe919d0c/media/docs/efficient_gemm.md#threadblock-rasterization) 以增强 L2 缓存重用

#### 全 JIT 设计 🐳

DeepGEMM 采用完全 [即时](deep_gemm/jit) (JIT) 设计，安装时无需编译。所有内核都在运行时使用轻量级 JIT 实现进行编译。这种方法提供了几个优点：

- GEMM 形状、块大小和流水线阶段数被视为编译时常量
  - 节省寄存器
  - 编译器可以进行更多优化
- 自动选择块大小、warpgroup 数量、最佳流水线阶段和 TMA 集群大小
  - 但无需自动调整，最佳选择是确定性地选择的
- 完全展开 MMA 流水线，为编译器提供更多优化机会
  - 对于小形状非常重要
  - 有关详细信息，请参阅 [内核文件](deep_gemm/include/deep_gemm/fp8_gemm.cuh) 中的 `launch_k_iterations`

总体而言，JIT 显著提高了小形状的性能，类似于 [Triton](https://github.com/triton-lang/triton/) 编译器的方法。

#### 非对齐块大小 🐳

对于某些形状，对齐到 2 的幂的块大小会导致 SM 利用率不足。例如，对于 `M=256, N=7168`，典型的块大小分配 `BLOCK_M=128, BLOCK_N=128` 仅利用 132 个 SM 中的 `(256 / 128) * (7168 / 128) = 112` 个。为了解决这个问题，我们支持非对齐块大小，如 112，在这种情况下，可以使 `(256 / 128) * (7168 / 112) = 128` 个 SM 工作。实现这项技术以及细粒度缩放需要仔细优化，但最终可以提高性能。

#### FFMA SASS 交错 🐳

我们观察到 [CUTLASS FP8 内核](https://github.com/NVIDIA/cutlass/tree/main/examples/54_hopper_fp8_warp_specialized_gemm) 在 NVCC 12.2 和 12.3 之间存在性能提升。通过比较编译后的 SASS，我们发现 [一系列 `FADD` 指令](https://github.com/NVIDIA/cutlass/blob/eefa171318b79cbe2e78514d4cce5cd0fe919d0c/include/cutlass/gemm/collective/fp8_accumulation.hpp#L73) 中的一位以交错模式翻转。
在参考了一些开源 [CUDA 汇编器](https://github.com/cloudcores/CuAssembler/blob/master/CuAsm/CuControlCode.py#L46) 实现后，我们确定该位控制 `yield`，这可能会增强 warp 级并行性（只是猜测，让出当前 warp 并让其他 warp 工作）。

为了利用这一点，我们开发了 [一个类似的脚本](deep_gemm/jit/interleave_ffma.py) 来修改编译后的二进制文件中的 `FFMA` 指令。除了简单地修改 `yield` 位之外，我们还翻转了 `reuse` 位（如果 warp 被让出，则寄存器无法重用）。通过为 MMA 指令与提升 `FFMA` 指令重叠创造更多机会，此调整提高了细粒度缩放 FP8 GEMMs 的性能（在某些情况下提高 10% 以上）。

## 致谢

DeepGEMM 的灵感来自 [CUTLASS](https://github.com/nvidia/cutlass) 项目。感谢并尊重开发者们！

## 许可证

此代码仓库在 [MIT 许可证](LICENSE) 下发布。

## 引用

```bibtex
@misc{deepgemm2025,
      title={DeepGEMM：具有细粒度缩放的干净且高效的 FP8 GEMM 内核},
      author={Chenggang Zhao and Liang Zhao and Jiashi Li and Zhean Xu},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/DeepGEMM}},
}
```
