<div align="center">
<h1><img src="static/images/triforce.png" height="40px" align="top"/> TriForce: Lossless Acceleration of Long Sequence <br> Generation with Hierarchical Speculative Decoding
</h1>


**trainig-free, accelerate long sequence generation**
</div>
<div align="center">
<b>Hanshi Sun</b><sup>1</sup>,
<b>Zhuoming Chen</b><sup>1</sup>,
<b>Xinyu Yang</b><sup>1</sup>,
<b>Yuandong Tian</b><sup>2</sup>,
<b>Beidi Chen</b><sup>1,2</sup>,
</div>

<div align="center">
<sup>1</sup>Carnegie Mellon University
<sup>2</sup>Meta AI (FAIR)
</div>

<div align="center">
[<a href="https://arxiv.org/abs/2404.XXXX">Paper</a>] | [<a href="https://infini-ai-lab.github.io/TriForce">Blog</a>]
</div>
<br>
<div align="center">
<img src="static/images/TriForce.gif" align="top"/>
<figcaption>serving <a href="https://huggingface.co/LargeWorldModel/LWM-Text-Chat-128K">LWM-Text-Chat-128K</a> with offloading on 2x RTX-4090 (prefill 127K contexts)</figcaption>
</div>
<br>

<div align="center">
<img src="static/images/sys.png" align="top"/>
<figcaption>TriForce Framework</figcaption>
</div>

## Environment Set Up
```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation # install flash-attn
```

## Evaluations
Currently, only long-context Llama models are supported (including [Llama2-7B-128K](https://huggingface.co/NousResearch/Yarn-Llama-2-7b-128k), [Llama2-13B-128K](https://huggingface.co/NousResearch/Yarn-Llama-2-13b-128k), [LWM-Text-128K](https://huggingface.co/LargeWorldModel/LWM-Text-128K), [LWM-Text-Chat-128K](https://huggingface.co/LargeWorldModel/LWM-Text-Chat-128K)).

### On-Chip
On-chip results can be reproduced on A100 by running the following command. `--prefill` specifies the context length of prompt and `--budget` specifies the budget of retrieval cache. `chunk_size` specifies the chunk size of the KV cache. `top_p` and `temp` are the sampling hyperparameters. `gamma` is the number of speculative decoding steps.

```bash
# w/ CUDA Graph
CUDA_VISIBLE_DEVICES=0 python test/on_chip.py --prefill 124928 --budget 4096 \
 --chunk_size 8 --top_p 0.9 --temp 0.6 --gamma 6
```

### Offloading
For the offloading setting, we provide two implementations: TriForce w/ [Seqouia](https://github.com/Infini-AI-Lab/Sequoia) and TriForce w/o Seqouia. The difference between them is that the former uses Seqouia for tree attention (sometimes Seqouia will be early terminated due to numerical instability of the model when calculating the probility residual, leading to inaccurate results), while the latter uses the vanilla attention. The performance of offloading significantly relies on bandwidth of PCIE. In order to get accurate results, it is best to ensure that the bandwidth is not used by other programs.

`--on_chip` specifies the number of layers' KV cache that are on-chip, which can adjusted based on hardware. The default tree size for Seqouia is set to 512.

```bash
# TriForce w/ Seqouia
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=48 torchrun --nproc_per_node=1 \
test/offloading_seqouia.py --budget 12288 --prefill 130048 \
--dataset demo --target llama-7B-128K --on_chip 9 --seed 1

# TriForce w/ CUDA Graph
CUDA_VISIBLE_DEVICES=0 python test/offloading.py --prefill 130048  \
--chunk_size 8 --temp 0.6 --top_p 0.9 --gamma 16 --dataset 128k \
--budget 0.1
```

### Offloading with Tensor Parallelism
Our framework supports tensor parallelism for offloading setting. The `--nproc_per_node` should be set to the number of GPUs used for offloading. The following command demonstrates how to use tensor parallelism with 2 GPUs. It should be noted that RTX-4090s do not support CUDA Graph for tensor parallelism (A100 supports). Therefore, we disabled CUDA Graph for this setting.

```bash
# TriForce w/ Seqouia w/o CUDA Graph
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=48 torchrun --nproc_per_node=2 \
test/offloading_seqouia.py --budget 12288 --prefill 130048 --dataset demo \
--target llama-7B-128K --on_chip 9 --seed 1

# TriForce w/o CUDA Graph
CUDA_VISIBLE_DEVICES=8,9 OMP_NUM_THREADS=48 torchrun --nproc_per_node=2 \
test/offloading_TP.py --budget 12288 --prefill 130048 --dataset demo \
--target llama-7B-128K --on_chip 9 --seed 1 --gamma 16
```

### Baseline
We provide the auto-regressive baseline implementation for comparison.

```bash
CUDA_VISIBLE_DEVICES=8,9 OMP_NUM_THREADS=48 torchrun --nproc_per_node=2 \
test/offloading_seqouia.py --prefill 130048 --dataset demo \
--target llama-7B-128K --on_chip 12 --seed 1 --baseline
```

## Citation
If you find TriForce useful or relevant to your project and research, please kindly cite our paper:

```bibtex
@article{sun2024triforce,
  title={TriForce: Lossless Acceleration of Long Sequence Generation with Hierarchical Speculative Decoding},
  author={Sun, Hanshi and Chen, Zhuoming and Yang, Xinyu and Tian, Yuandong and Chen, Beidi},
  journal={arXiv preprint arXiv:2404.XXXX},
  year={2024}
}
```
