import torch
import math
import time
import numpy as np
import os
import sys
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from utils.misc import spec_stream, log_csv
from utils.sampling import sample, norm_logits, max_fn

@torch.inference_mode()
def Autoregressive(tokenizer, graph_engine, input_ids, max_len=256, top_k=-1, top_p=0.9, temperature=0.6, verbose=False):
    # reset all cache
    graph_engine.engine.kv_cache.reset()

    logits = graph_engine.inference(input_ids=input_ids)

    if verbose:
        graph_engine.engine.kv_cache.print_status()

    next_token = sample(norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))
    
    if verbose:
        spec_stream(next_token[0], tokenizer, 'cyan')

    n = 0
    time1 = time.time()
    while n < max_len:
        logits = graph_engine.engine.model(input_ids=next_token, kv_cache=graph_engine.engine.kv_cache, graph_cache=None).logits
        next_token = sample(norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))
        n += 1
        if verbose:
            spec_stream(next_token[0], tokenizer, 'cyan')
    time2 = time.time()
    return n / (time2 - time1)


@torch.inference_mode()
def TriForce(tokenizer, graph_engine, input_ids, gamma=4, max_len=256, top_k=-1, top_p=0.9, temperature=0.6, verbose=False, file_path=None, dataset=None, spec_args=None):

    # reset all cache
    graph_engine.engine.kv_cache.reset()
    graph_engine.engine.graph_cache.reset()
    graph_engine.engine.draft_cache.reset()

    logits = graph_engine.inference(input_ids=input_ids[:,:-1])
    logits = graph_engine.inference(input_ids=input_ids[:,-1:])
    _ = graph_engine.graph_draft_prefill(input_ids=input_ids)

    if verbose:
        graph_engine.engine.kv_cache.print_status()
        graph_engine.engine.graph_cache.print_status()
        graph_engine.engine.draft_cache.print_status()

    resample_count = 0
    accepted_count = 0
    target_sample_count = 0
    draft_count = 0

    next_token = sample(norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))
    
    if verbose:
        spec_stream(next_token[0], tokenizer, 'cyan')

    acc_rate_middle_list = []
    n = 0
    time1 = time.time()
    while n < max_len:
        if next_token.shape == torch.Size([1]):
            next_token = next_token.unsqueeze(0)
        
        # speculative decoding for draft (68m) and retrieval 7b model
        pred_token_idx = next_token
        verify_tokens, speculation_probs, acc_rate_middle = Middle_Spec(pred_token_idx, graph_engine, gamma, False, tokenizer)
        acc_rate_middle_list.append(acc_rate_middle)
        generated_ids = verify_tokens[1:]
        draft_count += len(speculation_probs)

        gamma2 = len(generated_ids)
        
        # speculative decoding retrieval 7b model and target model
        verify_tokens = torch.cat([next_token, torch.LongTensor([generated_ids]).to(graph_engine.engine.model.device)], dim=1)
        logits = graph_engine.inference(input_ids=verify_tokens)

        count = 0
        verify_probs = []
    
        probs = norm_logits(logits[0], temperature=temperature ,top_k=top_k, top_p=top_p)
        for i in range(gamma2 + 1):
            verify_probs.append(probs[i])

        pass_tokens = torch.full((1, gamma2 + 2), 100, device=graph_engine.engine.model.device)
        pass_tokens[:, 0] = next_token
        
        for i, speculation_prob, verify_prob in zip(generated_ids, speculation_probs, verify_probs):
            r = torch.rand(1, device = graph_engine.engine.model.device)
            if r < torch.min(torch.tensor([1], device=r.device), (verify_prob[i] / speculation_prob[i])):
                count += 1
                accepted_count += 1
                n += 1
                pred_token_idx = torch.tensor([[i]]).to(graph_engine.engine.model.device)
                pass_tokens[:, count] = pred_token_idx
                if verbose:
                    spec_stream(i, tokenizer, 'green')
                # if eos
                if tokenizer.eos_token_id == i:
                    draft_count -= gamma2 - count
                    break
            else:
                resample_count += 1
                n += 1
                pred_token_idx = sample(max_fn(verify_prob-speculation_prob))
                pass_tokens[:, count+1] = pred_token_idx
                if verbose:
                    spec_stream(pred_token_idx, tokenizer, 'red')
                break

            if tokenizer.eos_token_id == pred_token_idx:
                break

        # update 7b cache
        graph_engine.engine.kv_cache.seq_len -= (len(generated_ids) - count)
        graph_engine.update_graph_cache()
        
        if count == len(generated_ids):
            target_sample_count += 1
            n += 1
            pred_token_idx = sample(verify_probs[-1])
            pass_tokens[:, count+1] = pred_token_idx
            if verbose:
                spec_stream(pred_token_idx, tokenizer, 'blue')
            count += 1

        # update cache for 68m
        graph_engine.graph_draft_inference(input_ids=pass_tokens, gamma_offset = gamma2 + 1)
        current_seq_len = graph_engine.engine.draft_cache.start_size + graph_engine.engine.draft_cache.recent_size + count
        graph_engine.engine.draft_cache.evict_for_spec(current_seq_len)

        next_token = pred_token_idx

    time2 = time.time()
    acceptance_rate = accepted_count / draft_count
    avg_tokens = accepted_count / draft_count * gamma
    if verbose:
        print(f"Use {time2 - time1} sec to generate {n} tokens (now {graph_engine.engine.kv_cache.seq_len} tokens), Tokens/s: {n / (time2 - time1)}", flush=True)
        print(f"accepted rate {acceptance_rate}, avg generated tokens {avg_tokens}")

    if file_path is not None:
        header = "target,acceptance_rate,token/s,avg_tokens,prefill,gen_len,dataset,acc_rate_middle,latency\n"
        entry = f"{graph_engine.engine.model.config._name_or_path},{acceptance_rate},{n / (time2 - time1)},{avg_tokens},{input_ids.shape[1]},{n},{dataset},{np.array(acc_rate_middle_list).mean()},{(time2 - time1)/n}\n"

        if spec_args is not None:
            for k, v in spec_args.items():
                header=header.replace("\n", f",{k}\n")
                entry=entry.replace("\n", f",{v}\n")
        log_csv(file_path, header, entry)

    return acceptance_rate, n / (time2 - time1)

@torch.inference_mode()
def Middle_Spec(next_token, graph_engine, gamma, verbose, tokenizer):

    n = 0
    resample_count = 0
    accepted_count = 0
    target_sample_count = 0
    draft_count = 0

    pred_token_idx = next_token

    return_generated_ids = []
    return_speculation_probs = []
    return_generated_ids.append(next_token.item())

    verify_tokens = torch.full((1, gamma + 1), 100, device=graph_engine.engine.model.device)
    verify_tokens[:, 0] = next_token

    position_ids = torch.arange(graph_engine.engine.kv_cache.seq_len, graph_engine.engine.kv_cache.seq_len+gamma+1, device=graph_engine.engine.model.device).unsqueeze(0)

    while n < gamma:
        speculation_prob = graph_engine.graph_draft_inference(input_ids=verify_tokens[:,:n+1], gamma_offset = n)
        
        pred_token_idx = sample(speculation_prob)
        token_idx = pred_token_idx.item()
        draft_count += 1

        verify_tokens[:, n+1:n+2] = pred_token_idx
        verify_prob = graph_engine.graph_verify(input_ids=verify_tokens, position_ids=position_ids)

        r = torch.rand(1, device = graph_engine.engine.model.device)
        if r < torch.min(torch.tensor([1], device=r.device), (verify_prob[n, token_idx] / speculation_prob[token_idx])):
            return_speculation_probs.append(verify_prob[n])
            return_generated_ids.append(token_idx)
            if verbose:
                spec_stream(pred_token_idx, tokenizer, 'green')
            accepted_count += 1
            n += 1
        
            pred_token_idx = sample(verify_prob[n])
            return_speculation_probs.append(verify_prob[n])
            return_generated_ids.append(pred_token_idx.item())
            if verbose:
                spec_stream(pred_token_idx, tokenizer, 'blue')
            target_sample_count += 1
            n += 1

            verify_tokens[:, n:n+1] = pred_token_idx
        
        else:
            pred_token_idx = sample(verify_prob[n])
            return_speculation_probs.append(verify_prob[n])
            return_generated_ids.append(pred_token_idx.item())
            if verbose:
                spec_stream(pred_token_idx, tokenizer, 'red')
            resample_count += 1
            n += 1

            verify_tokens[:, n:n+1] = pred_token_idx
    
    acceptance_rate = accepted_count / draft_count
    return return_generated_ids, return_speculation_probs, acceptance_rate



################### Dist Spec ####################
import torch.distributed as dist

def sample_dist(probs):
    if torch.distributed.get_rank() == 0:
        next_token = sample(probs)
    else:
        next_token = torch.empty((1,1), dtype=torch.long, device=probs.device)

    torch.distributed.broadcast(next_token, src=0)
    dist.barrier()
    # assert next_token.item() <= probs.shape[-1], f"{next_token.item()} > {probs.shape[-1]}, probs: {probs}"
    return next_token


@torch.inference_mode()
def Baseline_Dist(tokenizer, graph_engine, input_ids, max_len=256, top_k=-1, top_p=0.9, temperature=0.6, verbose=False, local_rank=0):
    bsz, prefill = input_ids.size()
    graph_engine.reset()
    logits = graph_engine.prefill(input_ids=input_ids)
    
    next_token = sample_dist(norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))
    
    gen_tokens = torch.zeros((input_ids.size(0), max_len), dtype=torch.long, device=input_ids.device)

    n = 0
    pos = 0
    generated_ids = []
    generated_ids.extend(next_token[0].tolist())
    
    torch.cuda.synchronize()
    time1 = time.time()
    while n < max_len:
        logits = graph_engine.inference(input_ids=next_token)
        
        next_token = sample_dist(norm_logits(logits[:,-1,:], temperature=temperature ,top_k=top_k, top_p=top_p))
        
        generated_ids.extend(next_token[0].tolist())

        generated_text = (
            tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            spaces_between_special_tokens=False,
        )
        .strip()
        .split(" ")
        )

        if local_rank == 0:
            now = len(generated_text) - 1
            if now > pos:
                print(" ".join(generated_text[pos:now]), end=" ", flush=True)
                pos = now

        gen_tokens[:, n] = next_token.squeeze()
        n += 1
    torch.cuda.synchronize()
    time2 = time.time()
    return 1000 * (time2 - time1) / n, gen_tokens


@torch.inference_mode()
def TriForce_Dist(tokenizer, llm, input_ids, gamma=4, max_len=256, top_k=-1, top_p=0.9, temperature=0.6, verbose=False, file_path=None, dataset=None, spec_args=None):

    ##### PREFILL #####
    llm.reset()
    llm.prefill(input_ids=input_ids[:,:-1])
    logits = llm.build_retrieval_cache(input_ids=input_ids[:,-1:])
    next_token = sample_dist(norm_logits(logits[:,-1,:], temperature=temperature ,top_k=-1, top_p=top_p))

    if next_token.shape == torch.Size([1]):
        next_token = next_token.unsqueeze(0)
    
    _ = llm.draft_run(input_ids=input_ids)

    resample_count = 0
    accepted_count = 0
    target_sample_count = 0
    draft_count = 0
    
    if verbose:
        spec_stream(next_token[0], tokenizer, 'cyan')

    acc_rate_middle_list = []
    n = 0

    pos = 0
    print_ids = []
    print_ids.extend(next_token[0].tolist())

    time1 = time.time()
    while n < max_len:
        if next_token.shape == torch.Size([1]):
            next_token = next_token.unsqueeze(0)
        
        # speculative decoding for draft (68m) and retrieval 7b model
        pred_token_idx = next_token
        verify_tokens, speculation_probs, acc_rate_middle = Middle_Spec_Dist(pred_token_idx, llm, gamma, False, tokenizer)
        acc_rate_middle_list.append(acc_rate_middle)
        generated_ids = verify_tokens[1:]
        draft_count += len(speculation_probs)

        gamma2 = len(generated_ids)
        
        # speculative decoding retrieval 7b model and target model
        verify_tokens = torch.cat([next_token, torch.LongTensor([generated_ids]).to(llm.device)], dim=1)
        logits = llm.inference(input_ids=verify_tokens)

        count = 0
        verify_probs = []
    
        probs = norm_logits(logits[0], temperature=temperature, top_k=top_k, top_p=top_p)
        for i in range(gamma2 + 1):
            verify_probs.append(probs[i])

        pass_tokens = torch.full((1, gamma2 + 2), 100, device=llm.device)
        pass_tokens[:, 0] = next_token
        
        for i, speculation_prob, verify_prob in zip(generated_ids, speculation_probs, verify_probs):
            r = torch.rand(1, device = llm.device)
            # broadcast the random number
            torch.distributed.broadcast(r, src=0)
            dist.barrier()
            if r <= torch.min(torch.tensor([1], device=r.device), (verify_prob[i] / speculation_prob[i])):
                count += 1
                accepted_count += 1
                n += 1
                pred_token_idx = torch.tensor([[i]]).to(llm.device)
                pass_tokens[:, count] = pred_token_idx
                if verbose:
                    spec_stream(i, tokenizer, 'green')
                print_ids.append(pred_token_idx.item())
                # if eos
                if tokenizer.eos_token_id == i:
                    draft_count -= gamma2 - count
                    if llm.local_rank == 0:
                        print('[EOS]')
                    break
            else:
                resample_count += 1
                n += 1
                pred_token_idx = sample_dist(max_fn(verify_prob-speculation_prob))
                pass_tokens[:, count+1] = pred_token_idx
                if verbose:
                    spec_stream(pred_token_idx, tokenizer, 'red')
                print_ids.append(pred_token_idx.item())
                break

            if tokenizer.eos_token_id == pred_token_idx:
                if llm.local_rank == 0:
                    print('[EOS]')
                break

        if tokenizer.eos_token_id == pred_token_idx:
            break

        # update 7b cache
        llm.kv_cache.seq_len -= (len(generated_ids) - count)
        llm.retrieval_cache.update_graph_cache(llm.kv_cache)
        
        if count == len(generated_ids):
            target_sample_count += 1
            n += 1
            pred_token_idx = sample_dist(verify_probs[-1])
            pass_tokens[:, count+1] = pred_token_idx
            if verbose:
                spec_stream(pred_token_idx, tokenizer, 'blue')
            print_ids.append(pred_token_idx.item())
            count += 1

        # update cache for 68m
        # print(pass_tokens,pred_token_idx, flush=True)
        llm.draft_run(input_ids=pass_tokens, gamma_offset = gamma2 + 1)
        current_seq_len =llm.draft_cache.start_size + llm.draft_cache.recent_size + count
        llm.draft_cache.evict_for_spec(current_seq_len)

        generated_text = (
            tokenizer.decode(
            print_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            spaces_between_special_tokens=False,
        )
        .strip()
        .split(" ")
        )

        if llm.local_rank == 0:
            now = len(generated_text) - 1
            if now > pos:
                print(" ".join(generated_text[pos:now]), end=" ", flush=True)
                pos = now

        next_token = pred_token_idx

    time2 = time.time()
    acceptance_rate = accepted_count / draft_count
    avg_tokens = accepted_count / draft_count * gamma

    return avg_tokens, (time2 - time1) / n


@torch.inference_mode()
def Middle_Spec_Dist(next_token, llm, gamma, verbose, tokenizer):
    n = 0
    resample_count = 0
    accepted_count = 0
    target_sample_count = 0
    draft_count = 0

    pred_token_idx = next_token

    return_generated_ids = []
    return_speculation_probs = []
    return_generated_ids.append(next_token.item())

    verify_tokens = torch.full((1, gamma + 1), 100, device=llm.device)
    verify_tokens[:, 0] = next_token

    position_ids = torch.arange(llm.kv_cache.seq_len, llm.kv_cache.seq_len+gamma+1, device=llm.device).unsqueeze(0)

    while n < gamma:
        speculation_prob = llm.draft_run(input_ids=verify_tokens[:,:n+1], gamma_offset = n)
        
        pred_token_idx = sample_dist(speculation_prob)
        token_idx = pred_token_idx.item()
        draft_count += 1

        verify_tokens[:, n+1:n+2] = pred_token_idx
        verify_prob = llm.retrieval_verify(input_ids=verify_tokens, position_ids=position_ids, temperature=llm.temperature, top_p=llm.top_p)

        r = torch.rand(1, device = llm.device)
        # broadcast the random number
        torch.distributed.broadcast(r, src=0)
        dist.barrier()
        if r < torch.min(torch.tensor([1], device=llm.device), (verify_prob[n, token_idx] / speculation_prob[token_idx])):
            return_speculation_probs.append(verify_prob[n])
            return_generated_ids.append(token_idx)
            if verbose:
                spec_stream(pred_token_idx, tokenizer, 'green')
            accepted_count += 1
            n += 1
        
            pred_token_idx = sample_dist(verify_prob[n])
            return_speculation_probs.append(verify_prob[n])
            return_generated_ids.append(pred_token_idx.item())
            if verbose:
                spec_stream(pred_token_idx, tokenizer, 'blue')
            target_sample_count += 1
            n += 1

            verify_tokens[:, n:n+1] = pred_token_idx
        
        else:
            pred_token_idx = sample_dist(verify_prob[n])
            return_speculation_probs.append(verify_prob[n])
            return_generated_ids.append(pred_token_idx.item())
            if verbose:
                spec_stream(pred_token_idx, tokenizer, 'red')
            resample_count += 1
            n += 1

            verify_tokens[:, n:n+1] = pred_token_idx

    acceptance_rate = accepted_count / draft_count
    
    return return_generated_ids, return_speculation_probs, acceptance_rate
