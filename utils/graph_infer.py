import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import torch
from typing import List, Optional, Tuple, Union
import gc
import math
from tqdm import tqdm

from .sampling import norm_logits

class InferenceEngine:
    def __init__(self, model, cache, graph_cache, draft, draft_cache) -> None:

        ###### 7B ######
        self.model = model
        self.model.eval()
        self.kv_cache = cache
        self.graph_cache = graph_cache

        ###### 68 MB ######
        self.draft = draft
        self.draft.eval()
        self.draft_cache = draft_cache

    @torch.inference_mode()
    def model_run(self, input_ids: torch.LongTensor):
        if input_ids.shape[-1] > 64: # prefill
            iter_prefill = math.ceil(input_ids.shape[1] / 128)
            for i in range(iter_prefill):
                logits = self.model(
                    input_ids=input_ids[:, i*128:(i+1)*128],
                    kv_cache=self.kv_cache,
                    graph_cache=None,
                ).logits
        else: # verification
            logits = self.model(input_ids=input_ids, kv_cache=self.kv_cache, graph_cache=self.graph_cache).logits
        return logits

    @torch.inference_mode()
    def draft_run(self, input_ids: torch.LongTensor, gamma_offset: int=0, probs=False, temperature=0.6, top_p=0.9):
        if input_ids.shape[-1] > 64: # prefill
            iter_prefill = math.ceil(input_ids.shape[1] / 64)
            for i in range(iter_prefill):
                self.draft_cache.evict_prefill(64)
                logits = self.draft(
                    input_ids=input_ids[:, i*64:(i+1)*64],
                    kv_cache=self.draft_cache,
                    graph_cache=None,
                ).logits
        else: # decoding
            logits = self.draft(input_ids=input_ids, kv_cache=self.draft_cache, graph_cache=self.draft_cache, gamma_offset=gamma_offset).logits

        if probs: # without top_p
            return norm_logits(logits[0], temperature=temperature, top_k=-1, top_p=top_p)[-1]
        return logits

    @torch.inference_mode()
    def model_verify(self, input_ids: torch.LongTensor, position_ids: Optional[torch.LongTensor]=None, probs=False, temperature=0.6, top_p=0.9):
        # graph verification (used for cuda graph capture)
        logits = self.model(input_ids=input_ids, kv_cache=self.kv_cache, graph_cache=self.graph_cache, position_ids=position_ids, spec=True).logits
        if probs: # without top_p
            # return torch.nn.functional.softmax(logits/0.6, dim=-1)[0]
            return norm_logits(logits[0], temperature=temperature, top_k=-1, top_p=top_p)
        return logits

    def clear_kv(self):
        self.kv_cache.reset()
        self.graph_cache.reset()
        self.draft_cache.reset()

def draft_run_capture_graph(engine :InferenceEngine, gamma_offset :int =0, mempool=None, n_warmups :int=3, probs=False, temperature=0.6, top_p=0.9):
    device = engine.draft.device
    
    # draft run is incremental decoding
    static_input_ids = torch.full((1, gamma_offset+1), 0, dtype=torch.long, device=device)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            static_logits = engine.draft_run(input_ids=static_input_ids, gamma_offset=gamma_offset, probs=probs, temperature=temperature, top_p=top_p)
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    print(f"[draft run] capturing graph for {gamma_offset} (probs={probs}, temp={temperature}, top_p={top_p})...")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        static_logits = engine.draft_run(input_ids=static_input_ids, gamma_offset=gamma_offset, probs=probs, temperature=temperature, top_p=top_p)
    
    def run(input_ids):
        static_input_ids.copy_(input_ids)
        graph.replay()
        return static_logits.clone()

    return run

def model_verify_capture_graph(engine :InferenceEngine, mempool=None, n_warmups :int=3, gamma:int=6, probs=False, temperature=0.6, top_p=0.9):
    device = engine.model.device
    
    # model_verify is verifying gamma tokens
    static_input_ids = torch.full((1, gamma+1), 0, dtype=torch.long, device=device)
    # static_storage_ids = torch.arange(gamma+1, device=device)
    static_position_ids = torch.arange(gamma+1, device=device).unsqueeze(0)
    
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            static_logits = engine.model_verify(input_ids=static_input_ids, position_ids=static_position_ids, probs=probs, temperature=temperature, top_p=top_p)
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    print(f"[model verify] capturing graph for spec len {gamma} (probs={probs}, temp={temperature}, top_p={top_p})...")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        static_logits = engine.model_verify(input_ids=static_input_ids, position_ids=static_position_ids, probs=probs, temperature=temperature, top_p=top_p)
    
    def run(input_ids, position_ids):
        static_input_ids.copy_(input_ids)
        # static_storage_ids.copy_(storage_ids)
        static_position_ids.copy_(position_ids)
        graph.replay()
        return static_logits.clone()

    return run

class GraphInferenceEngine:
    def __init__(self, model, cache, graph_cache, draft, draft_cache) -> None:

        self.engine = InferenceEngine(model, cache, graph_cache, draft, draft_cache)
        self.callables = {}
        self.mempool = None

    @torch.inference_mode()
    def initialize_cuda_graph(self, gamma=6, probs=False, temperature=0.6, top_p=0.9):
        gc.collect()
        self.mempool = torch.cuda.graphs.graph_pool_handle()
        


        for gamma_offset in range(gamma+3):
            self.callables[gamma_offset] = draft_run_capture_graph(
                                                engine=self.engine,
                                                gamma_offset=gamma_offset,
                                                mempool=self.mempool,
                                                n_warmups=3,
                                                probs=probs,
                                                temperature=temperature,
                                                top_p=top_p
                                            )

        self.callable_model_verify = model_verify_capture_graph(
                                        engine=self.engine,
                                        mempool=self.mempool,
                                        n_warmups=3,
                                        gamma=gamma,
                                        probs=probs,
                                        temperature=temperature,
                                        top_p=top_p
                                    )

        self.engine.clear_kv()

    def clear_kv(self):
        self.engine.clear_kv()

    @torch.inference_mode()
    def graph_draft_inference(self, input_ids: torch.LongTensor, gamma_offset: int=0):
        # draft run
        return self.callables[gamma_offset](input_ids)
    
    @torch.inference_mode()
    def graph_draft_prefill(self, input_ids: torch.LongTensor):
        # draft run
        logits = self.engine.draft_run(input_ids=input_ids)
        return logits

    @torch.inference_mode()
    def inference(self, input_ids: torch.LongTensor):
        # model run
        return self.engine.model_run(input_ids=input_ids)

    @torch.inference_mode()
    def graph_verify(self, input_ids: torch.LongTensor, position_ids: torch.LongTensor):
        # model verify
        return self.callable_model_verify(input_ids, position_ids)

    def init_graph_cache(self):
        self.engine.graph_cache.init_graph_cache(kv_cache=self.engine.kv_cache)

    def update_graph_cache(self):
        self.engine.graph_cache.update_graph_cache(kv_cache=self.engine.kv_cache)
