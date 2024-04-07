from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from models.cache import Cache
from transformers.models.llama.modeling_llama import(
    LlamaRMSNorm,
    LlamaConfig,
    PreTrainedModel,
    repeat_kv,
    ACT2FN
)

from transformers.modeling_outputs import CausalLMOutputWithPast

from flash_attn import flash_attn_with_kvcache

from .config_yarn import LlamaConfig
from models.cache import Cache

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_single(x, cos, sin, position_ids, unsqueeze_dim=1):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.

    # print(f"x: {x.shape}, cos: {cos.shape}, sin: {sin.shape}, position_ids: {position_ids.shape}")
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    # print(f"cos: {cos.shape}, sin: {sin.shape}")
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x,):
        return (
            self.cos_cached.to(dtype=x.dtype),
            self.sin_cached.to(dtype=x.dtype),
        )

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj

class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None, flash=False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self._init_rope()

    def _init_rope(self):
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Cache = None,
        graph_cache: Optional[Cache] = None,
        storage_ids: Optional[torch.LongTensor] = None,
        gamma_offset: int = -1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        cos, sin = self.rotary_emb(value_states)

        if gamma_offset >= 0: # graph spec
            key_states, value_states = graph_cache.spec_update(new_k_cache=key_states, new_v_cache=value_states, layer_idx=self.layer_idx, gamma_offset=gamma_offset)
            
            kv_seq_len = gamma_offset + graph_cache.start_size + graph_cache.recent_size + 1

            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)

            position_ids = torch.arange(graph_cache.real_budget-graph_cache.gamma-1, graph_cache.real_budget-graph_cache.gamma+gamma_offset, device=position_ids.device).unsqueeze(0)
            query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
            key_position_ids = torch.arange(kv_seq_len, device=position_ids.device).unsqueeze(0)
            key_states = apply_rotary_pos_emb_single(key_states, cos, sin, key_position_ids)
        
        else: # prefill
            kv_seq_len = key_states.shape[-3]
            kv_seq_len += kv_cache.seq_len
            
            key_states, value_states = kv_cache.update(key_states, value_states, layer_idx=self.layer_idx)
            # print(f"query_states: {query_states.shape}, key_states: {key_states.shape}, value_states: {cos.shape}, seq_len: {kv_cache.seq_len}")

            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)

            # print(f"query_states: {query_states.shape}, key_states: {key_states.shape}, value_states: {cos.shape}, position_ids: {position_ids}")

            query_states = apply_rotary_pos_emb_single(query_states, cos, sin, position_ids)
            key_position_ids = torch.arange(kv_seq_len, device=position_ids.device).unsqueeze(0)
            key_states = apply_rotary_pos_emb_single(key_states, cos, sin, key_position_ids)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = flash_attn_with_kvcache(q=query_states, k_cache=key_states, v_cache=value_states, softmax_scale=1/torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float16)), causal=True)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = (
            LlamaAttention(config=config, layer_idx=layer_idx)
        )

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Cache = None,
        graph_cache: Optional[Cache] = None,
        storage_ids: Optional[torch.LongTensor] = None,
        gamma_offset: int = -1,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
            graph_cache=graph_cache,
            storage_ids=storage_ids,
            gamma_offset=gamma_offset,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Cache = None,
        graph_cache: Optional[Cache] = None,
        storage_ids: Optional[torch.LongTensor] = None,
        gamma_offset: int = -1,
    ):
        batch_size, seq_length = input_ids.shape[:2]
        kv_cache_length = kv_cache.seq_len
        if position_ids is None:
            # for verification
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(kv_cache_length, seq_length + kv_cache_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        inputs_embeds = self.embed_tokens(input_ids)
        # attention_mask = _prepare_4d_causal_attention_mask(
        #     attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        # )

        hidden_states = inputs_embeds

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
                graph_cache=graph_cache,
                storage_ids=storage_ids,
                gamma_offset=gamma_offset,
            )

            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return hidden_states

class LlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Cache = None,
        graph_cache: Optional[Cache] = None,
        storage_ids: Optional[torch.LongTensor] = None,
        gamma_offset: int = -1,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
            graph_cache=graph_cache,
            storage_ids=storage_ids,
            gamma_offset=gamma_offset,
        )

        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float()


        return CausalLMOutputWithPast(
            logits=logits,
        )