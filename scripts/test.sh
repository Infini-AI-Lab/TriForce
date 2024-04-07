# TriForce w/ Seqouia 1xGPU
CUDA_VISIBLE_DEVICES=8 OMP_NUM_THREADS=48 torchrun --nproc_per_node=1 \
test/offloading_seqouia.py --budget 12288 --prefill 130048 \
--dataset demo --target llama-7B-128K --on_chip 0 --seed 1

# TriForce 1xGPU
CUDA_VISIBLE_DEVICES=8 python test/offloading.py --prefill 130048  \
--chunk_size 8 --temp 0.6 --top_p 0.9 --gamma 16 --dataset demo \
--budget 0.1

# TriForce w/ Seqouia 2xGPU
CUDA_VISIBLE_DEVICES=8,9 OMP_NUM_THREADS=48 torchrun --nproc_per_node=2 \
test/offloading_seqouia.py --budget 12288 --prefill 130048 --dataset demo \
--target llama-7B-128K --on_chip 9 --seed 1

# TriForce 2xGPU
CUDA_VISIBLE_DEVICES=8,9 OMP_NUM_THREADS=48 torchrun --nproc_per_node=2 \
test/offloading_TP.py --budget 12288 --prefill 130048 --dataset demo \
--target lwm-128K --on_chip 9 --seed 1 --gamma 16

# 2xGPU baseline
CUDA_VISIBLE_DEVICES=8,9 OMP_NUM_THREADS=48 torchrun --nproc_per_node=2 \
test/offloading_TP.py --budget 12288 --prefill 130048 --dataset demo \
--target lwm-128K --on_chip 12 --seed 1 --gamma 16 --baseline

# baseline
CUDA_VISIBLE_DEVICES=8,9 OMP_NUM_THREADS=48 torchrun --nproc_per_node=2 \
test/offloading_seqouia.py --prefill 130048 --dataset demo \
--target llama-7B-128K --on_chip 12 --seed 1 --baseline