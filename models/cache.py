from typing import Any, Dict, List, Optional, Tuple
from numpy import dtype
import torch
import math

class Cache:
    """
    Base, abstract class for all caches. The actual data structure is specific to each subclass.
    """
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Make sure to implement `update` in a subclass.")


############## Single GPU Cache ###############
class FlashSimpleCache(Cache):
    def __init__(self, model, max_budget=1024) -> None:
        self.seq_len = 0
        self.max_budget = max_budget

        self.hidden_size = model.config.hidden_size
        self.num_heads = model.config.num_key_value_heads
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers

        dtype = model.model.layers[0].self_attn.q_proj.weight.dtype

        self.key_cache = torch.zeros([self.layers, 1, self.max_budget, self.num_heads, self.head_dim], dtype=dtype).to(model.device)
        self.value_cache = torch.zeros([self.layers, 1, self.max_budget, self.num_heads, self.head_dim], dtype=dtype).to(model.device)

        self.scores = []

    def print_status(self):
        print("[Full Cache] Cached:", self.seq_len, "| Budget:", self.max_budget)
    
    def reset(self):
        self.seq_len = 0
        for i in range(self.layers):
            self.key_cache[i].zero_()
            self.value_cache[i].zero_()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.key_cache[layer_idx][:, self.seq_len : self.seq_len + key_states.shape[-3]] = key_states
        self.value_cache[layer_idx][:, self.seq_len : self.seq_len + value_states.shape[-3]] = value_states

        key = self.key_cache[layer_idx][:, :self.seq_len + value_states.shape[-3]]
        value = self.value_cache[layer_idx][:, :self.seq_len + value_states.shape[-3]]

        if layer_idx == self.layers-1:
            self.seq_len += key_states.shape[-3]

        return key, value

class OffloadingFlashSimpleCache(Cache):
    def __init__(self, model, max_budget=1024) -> None:
        self.seq_len = 0
        self.max_budget = max_budget

        self.hidden_size = model.config.hidden_size
        self.num_heads = model.config.num_key_value_heads
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers

        dtype = model.model.layers[0].self_attn.q_proj.weight.dtype
        self.device = model.device

        self.key_cache = torch.zeros([self.layers, 1, self.max_budget, self.num_heads, self.head_dim], dtype=dtype, device='cpu').pin_memory()
        self.value_cache = torch.zeros([self.layers, 1, self.max_budget, self.num_heads, self.head_dim], dtype=dtype, device='cpu').pin_memory()

        # init layer cache buffer on chip
        self.key_cache_buffer = torch.zeros([1, self.max_budget, self.num_heads, self.head_dim], dtype=dtype, device=self.device)
        self.value_cache_buffer = torch.zeros([1, self.max_budget, self.num_heads, self.head_dim], dtype=dtype, device=self.device)

        self.load_stream = torch.cuda.Stream(device=self.device)

    def print_status(self):
        print("[Offloading Flash Simple Cache] Cached Size:", self.seq_len, "| Budget:", self.max_budget)
    
    def reset(self):
        self.seq_len = 0
        for i in range(self.layers):
            self.key_cache[i].zero_()
            self.value_cache[i].zero_()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # copy incoming k v cache to cpu
        self.key_cache[layer_idx][:, self.seq_len : self.seq_len + key_states.shape[-3]] = key_states.cpu()
        self.value_cache[layer_idx][:, self.seq_len : self.seq_len + value_states.shape[-3]] = value_states.cpu()

        # copy k v cache to buffer
        self.key_cache_buffer.copy_(self.key_cache[layer_idx], non_blocking=True)
        self.value_cache_buffer.copy_(self.value_cache[layer_idx], non_blocking=True)
        
        key = self.key_cache_buffer[:, :self.seq_len + value_states.shape[-3]]
        value = self.value_cache_buffer[:, :self.seq_len + value_states.shape[-3]]

        if layer_idx == self.layers-1:
            self.seq_len += key_states.shape[-3]

        return key, value

class RetrievalCache(Cache):
    def __init__(self, model, max_budget=1024, prefill=1024, chunk_size=8, gamma=6) -> None:
        
        self.chunk_size = chunk_size
        self.prefill = prefill
        self.chunks = prefill // self.chunk_size
        self.select_sets = max_budget // self.chunk_size
        self.gamma = gamma
        self.max_budget = max_budget
        assert prefill % self.chunk_size == 0, f"prefill should be multiple of chunk_size, got {prefill} % {self.chunk_size}"
        assert max_budget % self.chunk_size == 0, f"max_budget should be multiple of chunk_size, got {max_budget} % {self.chunk_size}"

        self.real_budget = max_budget + gamma + 1

        self.hidden_size = model.config.hidden_size
        self.num_heads = model.config.num_key_value_heads
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers

        dtype = model.model.layers[0].self_attn.q_proj.weight.dtype

        self.key_cache = torch.zeros([self.layers, 1, self.real_budget, self.num_heads, self.head_dim], dtype=dtype).to(model.device)
        self.value_cache = torch.zeros([self.layers, 1, self.real_budget, self.num_heads, self.head_dim], dtype=dtype).to(model.device)

        self.init_graph = False

    def print_status(self):
        print("[Retrieval Cache] Budget:", self.max_budget, " | PreFill:", self.prefill, " | Chunk Size:", self.chunk_size, " | Chunks:", self.chunks, " | Select Sets:", self.select_sets)

    def init_graph_cache(self, kv_cache, query_states, layer_idx):

        # query_states: (bsz, 1, 32, head_dim) --> (bsz, 32, 1, head_dim)
        # key_cache: (bsz, seq_len, 32, head_dim) --> (bsz, 32, head_dim, seq_len)
        # print(query_states.shape, self.chunk_k[layer_idx].shape)

        assert 1 == query_states.shape[1], "query_states should be 1 for init"

        chunk_k = kv_cache.key_cache[layer_idx,:,:self.prefill].cuda().view(1, self.chunks, self.chunk_size, self.num_heads, self.head_dim).mean(dim=-3)
        
        # (bsz, 32, chunks)
        chunk_attn = torch.matmul(query_states.permute(0, 2, 1, 3), chunk_k.permute(0, 2, 3, 1)).squeeze(2)
        # (bsz, 32, select_sets) --> (bsz, select_sets, 32)
        _, topk_idx_rest = torch.topk(chunk_attn[:, :, 1:], k=self.select_sets-1, dim=-1)
        topk_idx_rest += 1
        topk_idx_first = torch.zeros((topk_idx_rest.shape[0], topk_idx_rest.shape[1], 1), device=topk_idx_rest.device, dtype=topk_idx_rest.dtype)
        topk_idx = torch.cat([topk_idx_first, topk_idx_rest], dim=-1)  # (bsz, 32, select_sets)
        expanded_index_tensor = topk_idx.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)

        # (bsz, prefill, 32, head_dim) --> (bsz, chunks, chunk_size, 32, head_dim) --> (bsz, chunks, 32, chunk_size, head_dim)
        key_ = kv_cache.key_cache[layer_idx][:, :self.prefill].reshape(1, self.chunks, self.chunk_size, self.num_heads, self.head_dim).cuda()
        key_ = key_.permute(0, 1, 3, 2, 4)
        result_tensor = torch.gather(key_, 1, expanded_index_tensor) # (bsz, select_sets, 32, chunk_size, head_dim)
        # (bsz, select_sets, 32, chunk_size, head_dim) --> (bsz, select_sets*chunk_size, 32, head_dim)
        self.key_cache[layer_idx][:,:self.max_budget] = result_tensor.permute(0, 1, 3, 2, 4).reshape(1, self.select_sets*self.chunk_size, self.num_heads, self.head_dim).clone()

        value_ = kv_cache.value_cache[layer_idx][:, :self.prefill].reshape(1, self.chunks, self.chunk_size, self.num_heads, self.head_dim).cuda()
        value_ = value_.permute(0, 1, 3, 2, 4)
        result_tensor = torch.gather(value_, 1, expanded_index_tensor)
        self.value_cache[layer_idx][:,:self.max_budget] = result_tensor.permute(0, 1, 3, 2, 4).reshape(1, self.select_sets*self.chunk_size, self.num_heads, self.head_dim).clone()

        if layer_idx == self.layers-1:
            self.init_graph = True

    def update_graph_cache(self, kv_cache=None):
        self.value_cache[:,:,self.max_budget-(kv_cache.seq_len-self.prefill):self.max_budget] = kv_cache.value_cache[:,:, self.prefill:kv_cache.seq_len].clone()
        self.key_cache[:,:,self.max_budget-(kv_cache.seq_len-self.prefill):self.max_budget] = kv_cache.key_cache[:,:, self.prefill:kv_cache.seq_len].clone()

    def update(self, new_k_cache :torch.Tensor, new_v_cache :torch.Tensor, layer_idx :int):

        self.key_cache[layer_idx][:, self.real_budget-self.gamma-1:] = new_k_cache.clone()
        self.value_cache[layer_idx][:, self.real_budget-self.gamma-1:] = new_v_cache.clone()

        return self.key_cache[layer_idx][:,:self.real_budget], self.value_cache[layer_idx][:,:self.real_budget]

    def update_graph_cache_retrieval(self, kv_cache, query_states, layer_idx):
        self.init_graph_cache(kv_cache, query_states, layer_idx)
        self.value_cache[layer_idx,:,self.max_budget-(kv_cache.seq_len-self.prefill):self.max_budget] = kv_cache.value_cache[layer_idx,:, self.prefill:kv_cache.seq_len].clone()
        self.key_cache[layer_idx,:,self.max_budget-(kv_cache.seq_len-self.prefill):self.max_budget] = kv_cache.key_cache[layer_idx,:, self.prefill:kv_cache.seq_len].clone()

    def reset(self):
        self.key_cache.zero_()
        self.value_cache.zero_()

class StreamingLLMEvictionCache(Cache):

    def __init__(self, model, gamma=6, start_size=16, recent_size=496) -> None:

        self.gamma = gamma
        self.start_size = start_size
        self.recent_size = recent_size
        self.real_budget = self.start_size + self.recent_size + self.gamma + 1 + 1 + 1

        self.seq_len = 0 # just for prefill usage

        self.hidden_size = model.config.hidden_size
        self.num_heads = model.config.num_key_value_heads
        self.head_dim = self.hidden_size // model.config.num_attention_heads
        self.layers = model.config.num_hidden_layers

        self.key_cache = torch.zeros([self.layers, 1, self.real_budget, self.num_heads, self.head_dim], dtype=torch.float16).to(model.device)
        self.value_cache = torch.zeros([self.layers, 1, self.real_budget, self.num_heads, self.head_dim], dtype=torch.float16).to(model.device)
    
    def print_status(self):
        print("[StreamingLLM Cache] Start Size:", self.start_size, "| Recent Size:", self.recent_size, "| Gamma:", self.gamma, "| Real Budget:", self.real_budget, "| Cached:", self.seq_len)

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int):
        
        incoming = key_states.shape[-3]

        assert self.seq_len + incoming <= self.start_size + self.recent_size
        self.key_cache[layer_idx][:, self.seq_len:self.seq_len + incoming] = key_states.clone()
        self.value_cache[layer_idx][:, self.seq_len:self.seq_len + incoming] = value_states.clone()

        key = self.key_cache[layer_idx][:, :self.seq_len + incoming]
        value = self.value_cache[layer_idx][:, :self.seq_len + incoming]

        if layer_idx == self.layers-1:
            self.seq_len += incoming
        return key, value

    def spec_update(self, new_k_cache :torch.Tensor, new_v_cache :torch.Tensor, layer_idx :int, gamma_offset=0):

        start = self.real_budget-self.gamma-3
        end = self.real_budget-self.gamma-3+new_k_cache.shape[-3]

        self.key_cache[layer_idx][:, start:end] = new_k_cache.clone()
        self.value_cache[layer_idx][:, start:end] = new_v_cache.clone()

        return self.key_cache[layer_idx][:,:end], self.value_cache[layer_idx][:,:end]

    def reset(self):
        for i in range(self.layers):
            self.key_cache[i].zero_()
            self.value_cache[i].zero_()

    def evict_prefill(self, incoming):
        # evict
        if self.seq_len + incoming <= self.start_size + self.recent_size:
            return
        for layer_idx in range(self.layers):
            size_keep = self.recent_size - incoming
            self.key_cache[layer_idx][:, self.start_size:self.start_size+size_keep] = self.key_cache[layer_idx][:, self.seq_len-size_keep:self.seq_len].clone()
            self.value_cache[layer_idx][:, self.start_size:self.start_size+size_keep] = self.value_cache[layer_idx][:, self.seq_len-size_keep:self.seq_len].clone()

        self.seq_len = self.start_size + self.recent_size - incoming

    def evict_for_spec(self, current_seq_len):
        self.key_cache[:,:,self.start_size:self.start_size+self.recent_size] = self.key_cache[:,:, current_seq_len-self.recent_size:current_seq_len].clone()
        self.value_cache[:,:, self.start_size:self.start_size+self.recent_size] = self.value_cache[:,:, current_seq_len-self.recent_size:current_seq_len].clone()

############## Dist Cache ###############
class DistributedSimpleCache(Cache):
    def __init__(self, config, max_budget=1024, device=None, on_chip_layers=0, ssl=0):
        self.config = config
        self.world_size = self.config.world_size
        self.local_rank = self.config.local_rank
        self.device  = device
        self.ssl = ssl
        self.ssl_cur = 0
        self.max_budget = max_budget
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_key_value_heads // self.world_size
        
        self.head_dim = self.hidden_size // self.config.num_attention_heads
        self.layers = self.config.num_hidden_layers
        
        self.seq_len = 0
        dtype=torch.float16
        self.on_chip_layers = on_chip_layers

        self.key_cache = torch.zeros([self.on_chip_layers, 1, self.max_budget, self.num_heads, self.head_dim], dtype=dtype, device=device)
        self.value_cache = torch.zeros([self.on_chip_layers, 1, self.max_budget, self.num_heads, self.head_dim], dtype=dtype, device=device)

        self.cpu_key_cache=torch.zeros([self.layers-self.on_chip_layers, 1, self.max_budget, self.num_heads, self.head_dim], dtype=dtype, device='cpu', pin_memory=True)
        self.cpu_value_cache=torch.zeros([self.layers-self.on_chip_layers, 1, self.max_budget, self.num_heads, self.head_dim], dtype=dtype, device='cpu', pin_memory=True)

    def print_status(self):
        print("Cached Size:", self.seq_len, "| Max Budget:", self.max_budget)

    def reset(self):
        self.seq_len = 0
        self.cpu_key_cache.zero_()
        self.cpu_value_cache.zero_()
        self.key_cache.zero_()
        self.value_cache.zero_()

    def normal_(self, seq_len=1024*127):
        self.seq_len = seq_len
        self.cpu_key_cache.normal_()
        self.cpu_value_cache.normal_()
        self.key_cache.normal_()
        self.value_cache.normal_()

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int,) -> Tuple[torch.Tensor, torch.Tensor]:
        assert layer_idx + 1 <= self.on_chip_layers, (layer_idx, self.on_chip_layers)
        self.key_cache[layer_idx][:, self.seq_len : self.seq_len + key_states.shape[1]] = key_states.clone()
        self.value_cache[layer_idx][:, self.seq_len : self.seq_len + value_states.shape[1]] = value_states.clone()

        key = self.key_cache[layer_idx][:, :self.seq_len + value_states.shape[1]]
        value = self.value_cache[layer_idx][:, :self.seq_len + value_states.shape[1]]

        return key, value

    def ssl_update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int,) -> Tuple[torch.Tensor, torch.Tensor]:
        assert layer_idx + 1 <= self.ssl, (layer_idx, self.ssl)
        self.key_cache[layer_idx][:, self.seq_len+self.ssl_cur : self.seq_len+self.ssl_cur + key_states.shape[1]] = key_states.clone()
        self.value_cache[layer_idx][:, self.seq_len+self.ssl_cur : self.seq_len+self.ssl_cur + value_states.shape[1]] = value_states.clone()

        key = self.key_cache[layer_idx][:, :self.seq_len+self.ssl_cur + value_states.shape[1]]
        value = self.value_cache[layer_idx][:, :self.seq_len+self.ssl_cur + value_states.shape[1]]

        if layer_idx == self.ssl - 1:
            self.ssl_cur += key_states.shape[1]

        return key, value

    def gather_kv_incremental(self, indices: list[int], offset:int):
        indices = [i + offset for i in indices]

        self.key_cache[:,:, offset:offset + len(indices)].copy_(self.key_cache[:,:, indices].clone(), non_blocking=True)
        self.value_cache[:,:, offset:offset + len(indices)].copy_(self.value_cache[:,:, indices].clone(), non_blocking=True)

        self.cpu_key_cache[:, :, offset:offset + len(indices)].copy_(self.cpu_key_cache[:, :, indices].clone(), non_blocking=True)
        self.cpu_value_cache[:, :, offset:offset + len(indices)].copy_(self.cpu_value_cache[:, :, indices].clone(), non_blocking=True)

        self.seq_len = offset + len(indices)
        self.ssl_cur = 0

    def copy_back_from_buffer(self, kv_buffer, layer_idx:int):
        self.cpu_key_cache[layer_idx-self.on_chip_layers][:,self.seq_len: kv_buffer.seq_len].copy_(kv_buffer.key_cache[:, self.seq_len: kv_buffer.seq_len], non_blocking=True)
        self.cpu_value_cache[layer_idx-self.on_chip_layers][:,self.seq_len: kv_buffer.seq_len].copy_(kv_buffer.value_cache[:, self.seq_len: kv_buffer.seq_len], non_blocking=True)

        if layer_idx == self.layers - 1:
            self.seq_len = kv_buffer.seq_len
            self.ssl_cur = 0

class DistributedKVCacheBuffer:
    def __init__(self, config, max_budget=1024, device=None) -> None:

        self.config = config
        self.max_budget = max_budget
        self.device = device
        self.dtype = torch.float16

        self.world_size = config.world_size
        self.local_rank = config.local_rank

        self.num_hidden_layers = config.num_hidden_layers
        self.num_heads = config.num_key_value_heads // self.world_size
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.key_cache = torch.zeros(1, self.max_budget, self.num_heads, self.head_dim, device=self.device,dtype=self.dtype)
        self.value_cache = torch.zeros(1, self.max_budget, self.num_heads, self.head_dim, device=self.device,dtype=self.dtype)
        self.seq_len = 0

    def copy_kv(self, kv_cache, layer_idx):
        on_chip_layers = kv_cache.on_chip_layers
        self.key_cache[:,:kv_cache.seq_len].copy_(kv_cache.cpu_key_cache[layer_idx-on_chip_layers][:,:kv_cache.seq_len], non_blocking=True)
        self.value_cache[:,:kv_cache.seq_len].copy_(kv_cache.cpu_value_cache[layer_idx-on_chip_layers][:,:kv_cache.seq_len], non_blocking=True)
        self.seq_len = kv_cache.seq_len

    def update(self, key_states :torch.Tensor, value_states :torch.Tensor, layer_idx :int):
        input_length = key_states.shape[1]
        self.key_cache[:,self.seq_len:self.seq_len + input_length] = key_states
        self.value_cache[:,self.seq_len:self.seq_len + input_length] = value_states
        self.seq_len += input_length
        return self.key_cache[:,:self.seq_len], self.value_cache[:,:self.seq_len]

class DistributedRetrievalCache_Seqouia:

    def __init__(self, config, max_budget=1024, device=None, prefill=1024, chunk_size=8, tree_size=128) -> None:

        self.config = config
        self.world_size = self.config.world_size
        self.local_rank = self.config.local_rank
        self.device  = device
        
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_key_value_heads // self.world_size
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = self.hidden_size // config.num_attention_heads
        self.layers = self.config.num_hidden_layers

        self.chunk_size = chunk_size
        self.prefill = prefill
        self.chunks = prefill // self.chunk_size
        self.select_sets = max_budget // self.chunk_size
        self.tree_size = tree_size
        self.max_budget = max_budget
        assert prefill % self.chunk_size == 0, f"prefill should be multiple of chunk_size, got {prefill} % {self.chunk_size}"
        assert max_budget % self.chunk_size == 0, f"max_budget should be multiple of chunk_size, got {max_budget} % {self.chunk_size}"
        self.real_budget = max_budget + tree_size
        self.init_graph = False
        self.device=device
        dtype=torch.float16
        self.key_cache=torch.zeros([self.layers, 1, self.real_budget, self.num_heads, self.head_dim], dtype=dtype).to(device)
        self.value_cache=torch.zeros([self.layers, 1, self.real_budget, self.num_heads, self.head_dim], dtype=dtype).to(device)

    def print_status(self):
        print("Budget:", self.max_budget, " | Real Budget:", self.real_budget, " | PreFill:", self.prefill, " | Chunk Size:", self.chunk_size, " | Chunks:", self.chunks, " | Select Sets:", self.select_sets)

    def init_graph_cache(self, kv_cache, query_states, layer_idx):

        if self.init_graph == True:
            raise ValueError("Graph is already initialized")
        assert 1 == query_states.shape[1], "query_states should be 1 for init"

        if hasattr(kv_cache, 'cpu_key_cache'):
            key_cache = kv_cache.key_cache[layer_idx]
            value_cache = kv_cache.value_cache[layer_idx]
        else:
            key_cache = kv_cache.key_cache
            value_cache = kv_cache.value_cache

        # chunk_k: (bsz, chunks, chunk_size, kv_heads, head_dim) --> (bsz, chunks, kv_heads, head_dim)
        chunk_k = key_cache[:,:self.prefill].view(1, self.chunks, self.chunk_size, self.num_heads, self.head_dim).mean(dim=-3)
        chunk_attn = torch.matmul(query_states.permute(0, 2, 1, 3), chunk_k.permute(0, 2, 3, 1)).squeeze(2) # (bsz, 32, chunks)
        _, topk_idx_rest = torch.topk(chunk_attn[:, :, 1:], k=self.select_sets-1, dim=-1) # (bsz, 32, select_sets) --> (bsz, select_sets, 32)
        topk_idx_rest += 1
        topk_idx_first = torch.zeros((topk_idx_rest.shape[0], topk_idx_rest.shape[1], 1), device=topk_idx_rest.device, dtype=topk_idx_rest.dtype)
        topk_idx = torch.cat([topk_idx_first, topk_idx_rest], dim=-1)  # (bsz, 32, select_sets)
        expanded_index_tensor = topk_idx.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)

        # (bsz, prefill, 32, head_dim) --> (bsz, chunks, chunk_size, 32, head_dim) --> (bsz, chunks, 32, chunk_size, head_dim)
        key_ = key_cache[:, :self.prefill].reshape(1, self.chunks, self.chunk_size, self.num_heads, self.head_dim)
        key_ = key_.permute(0, 1, 3, 2, 4)
        result_tensor = torch.gather(key_, 1, expanded_index_tensor) # (bsz, select_sets, 32, chunk_size, head_dim)
        # (bsz, select_sets, 32, chunk_size, head_dim) --> (bsz, select_sets*chunk_size, 32, head_dim)
        self.key_cache[layer_idx][:,:self.max_budget] = result_tensor.permute(0, 1, 3, 2, 4).reshape(1, self.select_sets*self.chunk_size, self.num_heads, self.head_dim).clone()

        value_ = value_cache[:, :self.prefill].reshape(1, self.chunks, self.chunk_size, self.num_heads, self.head_dim)
        value_ = value_.permute(0, 1, 3, 2, 4)
        result_tensor = torch.gather(value_, 1, expanded_index_tensor)
        self.value_cache[layer_idx][:,:self.max_budget] = result_tensor.permute(0, 1, 3, 2, 4).reshape(1, self.select_sets*self.chunk_size, self.num_heads, self.head_dim).clone()

        if layer_idx == self.layers-1:
            self.init_graph = True


    def update(self, key_states :torch.Tensor, value_states :torch.Tensor, layer_idx :int, storage_ids):
        input_length = len(storage_ids)
        assert input_length == key_states.shape[1]
        assert input_length == value_states.shape[1]

        self.key_cache[layer_idx].index_copy_(dim=1, index=storage_ids, source=key_states)
        self.value_cache[layer_idx].index_copy_(dim=1, index=storage_ids, source=value_states)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update_graph_cache(self, kv_cache=None):

        # on-chip layers
        on_chip_layers = kv_cache.on_chip_layers
        self.value_cache[:on_chip_layers,:,self.max_budget-(kv_cache.seq_len-self.prefill):self.max_budget].copy_(kv_cache.value_cache[:,:,self.prefill:kv_cache.seq_len], non_blocking=True)
        self.key_cache[:on_chip_layers,:,self.max_budget-(kv_cache.seq_len-self.prefill):self.max_budget].copy_(kv_cache.key_cache[:,:,self.prefill:kv_cache.seq_len], non_blocking=True)

        # cpu layers
        self.value_cache[on_chip_layers:,:,self.max_budget-(kv_cache.seq_len-self.prefill):self.max_budget].copy_(kv_cache.cpu_value_cache[:,:,self.prefill:kv_cache.seq_len], non_blocking=True)
        self.key_cache[on_chip_layers:,:,self.max_budget-(kv_cache.seq_len-self.prefill):self.max_budget].copy_(kv_cache.cpu_key_cache[:,:,self.prefill:kv_cache.seq_len], non_blocking=True)

    def reset(self):
        self.key_cache.zero_()
        self.value_cache.zero_()
        self.init_graph = False

    def normal_(self):
        self.key_cache.normal_()
        self.value_cache.normal_()

class DistributedRetrievalCache:
    def __init__(self, config, max_budget=1024, device=None, prefill=1024, chunk_size=8, gamma=6) -> None:

        self.config = config
        self.world_size = self.config.world_size
        self.local_rank = self.config.local_rank
        self.device  = device
        
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.num_key_value_heads // self.world_size
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = self.hidden_size // config.num_attention_heads
        self.layers = self.config.num_hidden_layers

        self.chunk_size = chunk_size
        self.prefill = prefill
        self.chunks = prefill // self.chunk_size
        self.select_sets = max_budget // self.chunk_size
        self.gamma = gamma
        self.max_budget = max_budget
        assert prefill % self.chunk_size == 0, f"prefill should be multiple of chunk_size, got {prefill} % {self.chunk_size}"
        assert max_budget % self.chunk_size == 0, f"max_budget should be multiple of chunk_size, got {max_budget} % {self.chunk_size}"
        self.real_budget = max_budget + gamma + 1
        self.init_graph = False
        self.device=device
        dtype=torch.float16
        self.key_cache=torch.zeros([self.layers, 1, self.real_budget, self.num_heads, self.head_dim], dtype=dtype).to(device)
        self.value_cache=torch.zeros([self.layers, 1, self.real_budget, self.num_heads, self.head_dim], dtype=dtype).to(device)

    def print_status(self):
        print("Budget:", self.max_budget, " | Real Budget:", self.real_budget, " | PreFill:", self.prefill, " | Chunk Size:", self.chunk_size, " | Chunks:", self.chunks, " | Select Sets:", self.select_sets)

    def init_graph_cache(self, kv_cache, query_states, layer_idx):

        if self.init_graph == True:
            raise ValueError("Graph is already initialized")
        assert 1 == query_states.shape[1], "query_states should be 1 for init"

        if hasattr(kv_cache, 'cpu_key_cache'):
            key_cache = kv_cache.key_cache[layer_idx]
            value_cache = kv_cache.value_cache[layer_idx]
        else:
            key_cache = kv_cache.key_cache
            value_cache = kv_cache.value_cache

        # chunk_k: (bsz, chunks, chunk_size, kv_heads, head_dim) --> (bsz, chunks, kv_heads, head_dim)
        chunk_k = key_cache[:,:self.prefill].view(1, self.chunks, self.chunk_size, self.num_heads, self.head_dim).mean(dim=-3)
        chunk_attn = torch.matmul(query_states.permute(0, 2, 1, 3), chunk_k.permute(0, 2, 3, 1)).squeeze(2) # (bsz, 32, chunks)
        _, topk_idx_rest = torch.topk(chunk_attn[:, :, 1:], k=self.select_sets-1, dim=-1) # (bsz, 32, select_sets) --> (bsz, select_sets, 32)
        topk_idx_rest += 1
        topk_idx_first = torch.zeros((topk_idx_rest.shape[0], topk_idx_rest.shape[1], 1), device=topk_idx_rest.device, dtype=topk_idx_rest.dtype)
        topk_idx = torch.cat([topk_idx_first, topk_idx_rest], dim=-1)  # (bsz, 32, select_sets)
        expanded_index_tensor = topk_idx.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.chunk_size, self.head_dim)

        # (bsz, prefill, 32, head_dim) --> (bsz, chunks, chunk_size, 32, head_dim) --> (bsz, chunks, 32, chunk_size, head_dim)
        key_ = key_cache[:, :self.prefill].reshape(1, self.chunks, self.chunk_size, self.num_heads, self.head_dim)
        key_ = key_.permute(0, 1, 3, 2, 4)
        result_tensor = torch.gather(key_, 1, expanded_index_tensor) # (bsz, select_sets, 32, chunk_size, head_dim)
        # (bsz, select_sets, 32, chunk_size, head_dim) --> (bsz, select_sets*chunk_size, 32, head_dim)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.key_cache[layer_idx][:,:self.max_budget].copy_(result_tensor.permute(0, 1, 3, 2, 4).reshape(1, self.select_sets*self.chunk_size, self.num_heads, self.head_dim))

        value_ = value_cache[:, :self.prefill].reshape(1, self.chunks, self.chunk_size, self.num_heads, self.head_dim)
        value_ = value_.permute(0, 1, 3, 2, 4)
        result_tensor = torch.gather(value_, 1, expanded_index_tensor)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.value_cache[layer_idx][:,:self.max_budget].copy_(result_tensor.permute(0, 1, 3, 2, 4).reshape(1, self.select_sets*self.chunk_size, self.num_heads, self.head_dim))

        if layer_idx == self.layers-1:
            self.init_graph = True


    def update(self, key_states :torch.Tensor, value_states :torch.Tensor, layer_idx :int):

        self.key_cache[layer_idx][:, self.real_budget-self.gamma-1:] = key_states.clone()
        self.value_cache[layer_idx][:, self.real_budget-self.gamma-1:] = value_states.clone()

        return self.key_cache[layer_idx][:,:self.real_budget], self.value_cache[layer_idx][:,:self.real_budget]

    def update_graph_cache(self, kv_cache=None):

        # on-chip layers
        on_chip_layers = kv_cache.on_chip_layers
        self.value_cache[:on_chip_layers,:,self.max_budget-(kv_cache.seq_len-self.prefill):self.max_budget].copy_(kv_cache.value_cache[:,:,self.prefill:kv_cache.seq_len], non_blocking=True)
        self.key_cache[:on_chip_layers,:,self.max_budget-(kv_cache.seq_len-self.prefill):self.max_budget].copy_(kv_cache.key_cache[:,:,self.prefill:kv_cache.seq_len], non_blocking=True)

        # cpu layers
        self.value_cache[on_chip_layers:,:,self.max_budget-(kv_cache.seq_len-self.prefill):self.max_budget].copy_(kv_cache.cpu_value_cache[:,:,self.prefill:kv_cache.seq_len], non_blocking=True)
        self.key_cache[on_chip_layers:,:,self.max_budget-(kv_cache.seq_len-self.prefill):self.max_budget].copy_(kv_cache.cpu_key_cache[:,:,self.prefill:kv_cache.seq_len], non_blocking=True)

    def reset(self):
        self.key_cache.zero_()
        self.value_cache.zero_()
        self.init_graph = False

    def normal_(self):
        self.key_cache.normal_()
        self.value_cache.normal_()