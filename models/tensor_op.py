import torch
import torch.nn.functional as F
import math
from flash_attn import flash_attn_with_kvcache
import torch.distributed as dist
from typing import List, Optional, Tuple, Union

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def RMSNorm(
        hidden_states: torch.FloatTensor,
        layernorm_variance_epsilon: float,
        layernorm_weight: torch.FloatTensor
    ) -> torch.Tensor:

    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + layernorm_variance_epsilon)
    hidden_states = layernorm_weight * hidden_states.to(input_dtype)
    
    return hidden_states

def Attention(
    hidden_states: torch.FloatTensor,
    attention_mask: torch.FloatTensor,
    position_ids: torch.LongTensor,
    layer_idx: int,
    wq: torch.FloatTensor,
    wk: torch.FloatTensor,
    wv: torch.FloatTensor,
    wo: torch.FloatTensor,
    sin_cache: torch.FloatTensor,
    cos_cache: torch.FloatTensor,
    kv_cache,
    hidden_size: int,
    num_heads: int,
    num_key_value_heads: int,
    num_key_value_groups: int,
    head_dim: int
):
    bsz, q_len, _ = hidden_states.size()

    query_states = F.linear(hidden_states, wq)
    key_states = F.linear(hidden_states, wk)
    value_states = F.linear(hidden_states, wv)

    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    kv_seq_len += kv_cache.kv_offset

    cos = cos_cache[:kv_seq_len].to(value_states.dtype)
    sin = sin_cache[:kv_seq_len].to(value_states.dtype)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        
    key_states, value_states = kv_cache.update_kv_cache(key_states, value_states, layer_idx)
        
    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)
        
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

    attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, hidden_size)
    hidden_states = F.linear(attn_output, wo)

    return hidden_states

def TP_Attention(
    hidden_states: torch.FloatTensor,
    position_ids: torch.LongTensor,
    layer_idx: int,
    wq: torch.FloatTensor,
    wk: torch.FloatTensor,
    wv: torch.FloatTensor,
    wo: torch.FloatTensor,
    sin_cache: torch.FloatTensor,
    cos_cache: torch.FloatTensor,
    kv_buffer,
    hidden_size: int,
    local_num_heads: int,
    local_num_key_value_heads: int,
    num_key_value_groups: int,
    head_dim: int,
    attention_mask: torch.FloatTensor=None,
    flash_attn: bool=True,
    retrieval_cache=None
):
    bsz, q_len, _ = hidden_states.size()

    query_states = F.linear(hidden_states, wq) #[bsz, q_len, h // tp]
    key_states = F.linear(hidden_states, wk) #[bsz, q_len, h // tp]
    value_states = F.linear(hidden_states, wv) #[bsz, q_len, h // tp]

    query_states = query_states.view(bsz, q_len, local_num_heads, head_dim).transpose(1, 2)
    #[bsz, local_num_heads, q_len, head_dim]
    key_states = key_states.view(bsz, q_len, local_num_key_value_heads, head_dim).transpose(1, 2)
    #[bsz, local_num_kv_heads, q_len, head_dim]
    value_states = value_states.view(bsz, q_len, local_num_key_value_heads, head_dim)
    #[bsz, q_len, local_num_kv_heads, head_dim]

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos_cache, sin_cache, position_ids)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)

    key_states, value_states = kv_buffer.update(key_states, value_states, layer_idx)

    if retrieval_cache is not None:
        retrieval_cache.init_graph_cache(kv_buffer, query_states, layer_idx)

    if attention_mask is None:
        if bsz > 1:
            attn_output = flash_attn_with_kvcache(q=query_states, k_cache=key_states, v_cache=value_states, cache_seqlens=kv_buffer.seq_len, softmax_scale=1/torch.sqrt(torch.tensor(head_dim, dtype=torch.float16)), causal=True)
        else:
            attn_output = flash_attn_with_kvcache(q=query_states, k_cache=key_states, v_cache=value_states, softmax_scale=1/torch.sqrt(torch.tensor(head_dim, dtype=torch.float16)), causal=True)
    else:
        with torch.backends.cuda.sdp_kernel(enable_math=False):
            attn_output = F.scaled_dot_product_attention(query_states.transpose(1, 2),key_states.transpose(1, 2),value_states.transpose(1, 2), attn_mask=attention_mask.half())
        attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, local_num_heads * head_dim)
    #[bsz, q_len, h // tp]
    hidden_states = F.linear(attn_output, wo)

    #[bsz, q_len, h]
    dist.all_reduce(hidden_states, dist.ReduceOp.SUM)
    
    return hidden_states

def TP_Attention_ssl(
    hidden_states: torch.FloatTensor,
    position_ids: torch.LongTensor,
    layer_idx: int,
    wq: torch.FloatTensor,
    wk: torch.FloatTensor,
    wv: torch.FloatTensor,
    wo: torch.FloatTensor,
    sin_cache: torch.FloatTensor,
    cos_cache: torch.FloatTensor,
    kv_buffer,
    hidden_size: int,
    local_num_heads: int,
    local_num_key_value_heads: int,
    num_key_value_groups: int,
    head_dim: int,
    attention_mask: torch.FloatTensor=None,
    flash_attn: bool=True,
):
    bsz, q_len, _ = hidden_states.size()
    query_states = F.linear(hidden_states, wq) #[bsz, q_len, h // tp]
    key_states = F.linear(hidden_states, wk) #[bsz, q_len, h // tp]
    value_states = F.linear(hidden_states, wv) #[bsz, q_len, h // tp]
    query_states = query_states.view(bsz, q_len, local_num_heads, head_dim).transpose(1, 2)
    #[bsz, local_num_heads, q_len, head_dim]
    key_states = key_states.view(bsz, q_len, local_num_key_value_heads, head_dim).transpose(1, 2)
    #[bsz, local_num_kv_heads, q_len, head_dim]
    value_states = value_states.view(bsz, q_len, local_num_key_value_heads, head_dim)
    #[bsz, q_len, local_num_kv_heads, head_dim]
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos_cache, sin_cache, position_ids)
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    key_states, value_states = kv_buffer.ssl_update(key_states, value_states, layer_idx)
    with torch.backends.cuda.sdp_kernel(enable_math=False):
        attn_output = F.scaled_dot_product_attention(query_states.transpose(1, 2),key_states.transpose(1, 2),value_states.transpose(1, 2), attn_mask=attention_mask.half())
        attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, local_num_heads * head_dim)
    #[bsz, q_len, h // tp]
    hidden_states = F.linear(attn_output, wo)

    #[bsz, q_len, h]
    dist.all_reduce(hidden_states, dist.ReduceOp.SUM)
    
    return hidden_states


def TP_Attention_Tree_Retrieval(
    hidden_states: torch.FloatTensor,
    position_ids: torch.LongTensor,
    layer_idx: int,
    wq: torch.FloatTensor,
    wk: torch.FloatTensor,
    wv: torch.FloatTensor,
    wo: torch.FloatTensor,
    sin_cache: torch.FloatTensor,
    cos_cache: torch.FloatTensor,
    hidden_size: int,
    local_num_heads: int,
    local_num_key_value_heads: int,
    num_key_value_groups: int,
    head_dim: int,
    attention_mask: torch.FloatTensor=None,
    flash_attn: bool=True,
    retrieval_cache=None,
    storage_ids=None,
):
    bsz, q_len, _ = hidden_states.size()

    query_states = F.linear(hidden_states, wq) #[bsz, q_len, h // tp]
    key_states = F.linear(hidden_states, wk) #[bsz, q_len, h // tp]
    value_states = F.linear(hidden_states, wv) #[bsz, q_len, h // tp]

    query_states = query_states.view(bsz, q_len, local_num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, local_num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, local_num_key_value_heads, head_dim)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos_cache, sin_cache, position_ids)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    key_states, value_states = retrieval_cache.update(key_states=key_states, value_states=value_states, layer_idx=layer_idx, storage_ids=storage_ids)
    with torch.backends.cuda.sdp_kernel(enable_math=False):
        attn_output = F.scaled_dot_product_attention(query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2), attn_mask=attention_mask.half())
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, local_num_heads * head_dim)
    #[bsz, q_len, h // tp]
    hidden_states = F.linear(attn_output, wo)
    #[bsz, q_len, h]
    dist.all_reduce(hidden_states, dist.ReduceOp.SUM)
    return hidden_states



def TP_Attention_Retrieval(
    hidden_states: torch.FloatTensor,
    position_ids: torch.LongTensor,
    layer_idx: int,
    wq: torch.FloatTensor,
    wk: torch.FloatTensor,
    wv: torch.FloatTensor,
    wo: torch.FloatTensor,
    sin_cache: torch.FloatTensor,
    cos_cache: torch.FloatTensor,
    hidden_size: int,
    local_num_heads: int,
    local_num_key_value_heads: int,
    num_key_value_groups: int,
    head_dim: int,
    attention_mask: torch.FloatTensor=None,
    flash_attn: bool=True,
    retrieval_cache=None,
):
    bsz, q_len, _ = hidden_states.size()

    query_states = F.linear(hidden_states, wq) #[bsz, q_len, h // tp]
    key_states = F.linear(hidden_states, wk) #[bsz, q_len, h // tp]
    value_states = F.linear(hidden_states, wv) #[bsz, q_len, h // tp]

    query_states = query_states.view(bsz, q_len, local_num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, local_num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, local_num_key_value_heads, head_dim)

    cos = cos_cache.to(value_states.dtype)
    sin = sin_cache.to(value_states.dtype)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)

    key_states, value_states = retrieval_cache.update(key_states=key_states, value_states=value_states, layer_idx=layer_idx)

    if flash_attn:
        attn_output = flash_attn_with_kvcache(q=query_states, k_cache=key_states, v_cache=value_states, softmax_scale=1/torch.sqrt(torch.tensor(head_dim, dtype=torch.float16)), causal=True)
    else:
        raise ValueError("Non-Flash-Attn Retrieval TP-Attention is not implemented yet")

    attn_output = attn_output.reshape(bsz, q_len, local_num_heads * head_dim)

    #[bsz, q_len, h // tp]
    hidden_states = F.linear(attn_output, wo)

    #[bsz, q_len, h]
    dist.all_reduce(hidden_states, dist.ReduceOp.SUM)
    
    return hidden_states


def MLP(
    hidden_states: torch.FloatTensor,
    up_proj: torch.FloatTensor,
    down_proj: torch.FloatTensor,
    gate_proj: torch.FloatTensor,
    ):

    up = F.linear(hidden_states, up_proj)
    gate = F.linear(hidden_states, gate_proj)
    gate = F.silu(gate)
    hidden_states = gate * up
    hidden_states = F.linear(hidden_states, down_proj)

    return hidden_states

def TP_MLP(
    hidden_states: torch.FloatTensor,
    up_proj: torch.FloatTensor,
    down_proj: torch.FloatTensor,
    gate_proj: torch.FloatTensor,
    ):

    up = F.linear(hidden_states, up_proj)
    gate = F.linear(hidden_states, gate_proj)
    gate = F.silu(gate)
    hidden_states = gate * up
    hidden_states = F.linear(hidden_states, down_proj)
    
    dist.all_reduce(hidden_states, dist.ReduceOp.SUM)
    return hidden_states