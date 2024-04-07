# on-chip, single A100
CUDA_VISIBLE_DEVICES=0 python test/on_chip.py --prefill 124928 --budget 4096 --chunk_size 8 --top_p 0.9 --temp 0.6 --gamma 6 --dataset 128k

# offloading, single 4090/L40 (w/o Seqouia)
CUDA_VISIBLE_DEVICES=0 python test/offloading.py --prefill 130048  --chunk_size 8 --temp 0.6 --top_p 0.9 --gamma 16 --verbose --dataset gs --budget 0.1

# offloading, single 4090/L40 (w/ Seqouia)
CUDA_VISIBLE_DEVICES=0 python test/offloading_seqouia.py --prefill 130048  --chunk_size 8 --temp 0.6 --top_p 0.9 --gamma 16 --verbose --dataset gs --budget 0.1



# offloading, 2x4090, TP
