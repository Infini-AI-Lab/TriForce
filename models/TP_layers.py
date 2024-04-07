import torch
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from .config_yarn import LlamaConfig

class DistributedOffloadingConfig:
    def __init__(self, config: LlamaConfig, local_rank=0, world_size=1) -> None:

        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers 
        self.intermediate_size = config.intermediate_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.local_rank = local_rank
        self.world_size = world_size
        self.max_length = config.max_position_embeddings
        self.vocab_size = config.vocab_size

class LlamaLayer:
    def __init__(self, layer_idx) -> None:
        
        self.wq :torch.Tensor = None
        self.wk :torch.Tensor = None
        self.wv :torch.Tensor = None
        self.wo :torch.Tensor = None

        self.gate_proj :torch.Tensor = None 
        self.up_proj :torch.Tensor = None
        self.down_proj :torch.Tensor = None

        self.input_layernorm_weight :torch.Tensor = None
        self.input_layernorm_variance_epsilon :float = 0.0

        self.post_attention_layernorm_weight :torch.Tensor = None
        self.post_attention_layernorm_variance_epsilon :float = 0.0
        self.layer_idx = layer_idx
    
    def init_parameters(self, hf_layer: LlamaDecoderLayer):

        self.wq :torch.Tensor= hf_layer.self_attn.q_proj.weight.detach().pin_memory()
        self.wk :torch.Tensor= hf_layer.self_attn.k_proj.weight.detach().pin_memory()
        self.wv :torch.Tensor= hf_layer.self_attn.v_proj.weight.detach().pin_memory()
        self.wo :torch.Tensor= hf_layer.self_attn.o_proj.weight.detach().pin_memory()

        self.gate_proj = hf_layer.mlp.gate_proj.weight.detach().pin_memory()
        self.up_proj = hf_layer.mlp.up_proj.weight.detach().pin_memory()
        self.down_proj = hf_layer.mlp.down_proj.weight.detach().pin_memory()

        self.input_layernorm_weight = hf_layer.input_layernorm.weight.detach()
        self.input_layernorm_variance_epsilon = hf_layer.input_layernorm.variance_epsilon

        self.post_attention_layernorm_weight = hf_layer.post_attention_layernorm.weight.detach()
        self.post_attention_layernorm_variance_epsilon = hf_layer.post_attention_layernorm.variance_epsilon

        self.cos_cache :torch.Tensor= hf_layer.self_attn.rotary_emb.cos_cached
        self.sin_cache :torch.Tensor= hf_layer.self_attn.rotary_emb.sin_cached
    
    def init_gpu(self, device:str = 'cuda:0'):

        self.input_layernorm_weight = self.input_layernorm_weight.to(device)
        self.post_attention_layernorm_weight = self.post_attention_layernorm_weight.to(device)

        self.cos_cache = self.cos_cache.to(device)
        self.sin_cache = self.sin_cache.to(device)


class LlamaLayerBuffer:
    def __init__(self, device:str = 'cuda:0') -> None:
        self.device = device
    
    def init_space(self, layer: LlamaLayer):

        self.wq_buffer = torch.zeros_like(layer.wq).to(self.device)
        self.wk_buffer = torch.zeros_like(layer.wk).to(self.device)
        self.wv_buffer = torch.zeros_like(layer.wv).to(self.device)
        self.wo_buffer = torch.zeros_like(layer.wo).to(self.device)


        self.gate_proj_buffer = torch.zeros_like(layer.gate_proj).to(self.device)
        self.up_proj_buffer = torch.zeros_like(layer.up_proj).to(self.device)
        self.down_proj_buffer = torch.zeros_like(layer.down_proj).to(self.device)
    
    def sync_copy(self, layer: LlamaLayer):

        self.wq_buffer.copy_(layer.wq, non_blocking=True)
        self.wk_buffer.copy_(layer.wk, non_blocking=True)
        self.wv_buffer.copy_(layer.wv, non_blocking=True)
        self.wo_buffer.copy_(layer.wo, non_blocking=True)

        self.gate_proj_buffer.copy_(layer.gate_proj, non_blocking=True)
        self.up_proj_buffer.copy_(layer.up_proj, non_blocking=True)
        self.down_proj_buffer.copy_(layer.down_proj, non_blocking=True)

class DistributedLlamaLayer:
    def __init__(self, layer_idx, config: DistributedOffloadingConfig) -> None:
        
        self.wq :torch.Tensor = None
        self.wk :torch.Tensor = None
        self.wv :torch.Tensor = None
        self.wo :torch.Tensor = None

        self.gate_proj :torch.Tensor = None 
        self.up_proj :torch.Tensor = None
        self.down_proj :torch.Tensor = None

        self.input_layernorm_weight :torch.Tensor = None
        self.input_layernorm_variance_epsilon :float = 0.0

        self.post_attention_layernorm_weight :torch.Tensor = None
        self.post_attention_layernorm_variance_epsilon :float = 0.0

        self.layer_idx = layer_idx
        self.local_rank = config.local_rank
        self.world_size = config.world_size

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.world_size

        self.intermediate_size = config.intermediate_size
        self.mlp_slice = self.intermediate_size // self.world_size

    
    def init_parameters(self, hf_layer: LlamaDecoderLayer):

        self.wq :torch.Tensor= hf_layer.self_attn.q_proj.weight.detach()
        self.wq :torch.Tensor= self.wq.split((self.num_heads * self.head_dim) // self.world_size, dim=0)[self.local_rank].pin_memory()

        self.wk :torch.Tensor= hf_layer.self_attn.k_proj.weight.detach()
        self.wk :torch.Tensor= self.wk.split(self.key_value_slicing, dim=0)[self.local_rank].pin_memory()

        self.wv :torch.Tensor= hf_layer.self_attn.v_proj.weight.detach()
        self.wv :torch.Tensor= self.wv.split(self.key_value_slicing, dim=0)[self.local_rank].pin_memory()

        self.wo :torch.Tensor= hf_layer.self_attn.o_proj.weight.detach()
        self.wo :torch.Tensor=self.wo.split(self.hidden_size // self.world_size, dim=1)[self.local_rank].pin_memory()

        self.gate_proj :torch.Tensor= hf_layer.mlp.gate_proj.weight.detach()
        self.gate_proj :torch.Tensor = self.gate_proj.split(self.mlp_slice, dim=0)[self.local_rank].pin_memory()

        self.up_proj :torch.Tensor= hf_layer.mlp.up_proj.weight.detach()
        self.up_proj :torch.Tensor= self.up_proj.split(self.mlp_slice, dim=0)[self.local_rank].pin_memory()

        self.down_proj :torch.Tensor= hf_layer.mlp.down_proj.weight.detach().pin_memory()
        self.down_proj :torch.Tensor= self.down_proj.split(self.mlp_slice, dim=1)[self.local_rank].pin_memory()

        self.input_layernorm_weight = hf_layer.input_layernorm.weight.detach()
        if hasattr(hf_layer.input_layernorm, 'variance_epsilon'):
            self.input_layernorm_variance_epsilon = hf_layer.input_layernorm.variance_epsilon
        elif hasattr(hf_layer.input_layernorm, 'eps'):
            self.input_layernorm_variance_epsilon = hf_layer.input_layernorm.eps
        else:
            print("Error: variance_epsilon not found in input_layernorm")

        self.post_attention_layernorm_weight = hf_layer.post_attention_layernorm.weight.detach()
        if hasattr(hf_layer.post_attention_layernorm, 'variance_epsilon'):
            self.post_attention_layernorm_variance_epsilon = hf_layer.post_attention_layernorm.variance_epsilon
        elif hasattr(hf_layer.post_attention_layernorm, 'eps'):
            self.post_attention_layernorm_variance_epsilon = hf_layer.post_attention_layernorm.eps
        else:
            print("Error: variance_epsilon not found in post_attention_layernorm")

    def init_gpu(self, device:str = 'cuda:0'):

        self.input_layernorm_weight = self.input_layernorm_weight.to(device)
        self.post_attention_layernorm_weight = self.post_attention_layernorm_weight.to(device)

    def to_gpu(self, device:str = 'cuda:0'):

        self.wq = self.wq.to(device)
        self.wk = self.wk.to(device)
        self.wv = self.wv.to(device)
        self.wo = self.wo.to(device)

        self.gate_proj = self.gate_proj.to(device)
        self.up_proj = self.up_proj.to(device)
        self.down_proj = self.down_proj.to(device)


class DistributedLlamaLayerBuffer:
    def __init__(self, config:DistributedOffloadingConfig) -> None:
        self.device = torch.device("cuda", config.local_rank)
        self.config = config

    def init_space(self, layer: DistributedLlamaLayer):

        self.wq = torch.zeros_like(layer.wq).to(self.device)
        self.wk = torch.zeros_like(layer.wk).to(self.device)
        self.wv = torch.zeros_like(layer.wv).to(self.device)
        self.wo = torch.zeros_like(layer.wo).to(self.device)

        self.gate_proj = torch.zeros_like(layer.gate_proj).to(self.device)
        self.up_proj = torch.zeros_like(layer.up_proj).to(self.device)
        self.down_proj = torch.zeros_like(layer.down_proj).to(self.device)
    
    def sync_copy(self, layer: DistributedLlamaLayer):

        self.wq.copy_(layer.wq, non_blocking=True)
        self.wk.copy_(layer.wk, non_blocking=True)
        self.wv.copy_(layer.wv, non_blocking=True)
        self.wo.copy_(layer.wo, non_blocking=True)

        self.gate_proj.copy_(layer.gate_proj, non_blocking=True)
        self.up_proj.copy_(layer.up_proj, non_blocking=True)
        self.down_proj.copy_(layer.down_proj, non_blocking=True)