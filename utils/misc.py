from sympy import symbols, Eq, solve
from termcolor import colored
import random

def spec_stream(pred_token_idx, tokenizer, color='blue'):
    decoded_token = tokenizer.decode(
            pred_token_idx,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            # spaces_between_special_tokens=False,
        )

    decoded_token = decoded_token.replace("<0x0A>", "\n")

    print(colored(decoded_token, color), flush=True, end=" ")

def batch_spec_stream(pred_token_idx, tokenizer):
    color_list = ['red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'light_grey', 'dark_grey', 'light_red', 'light_green', 'light_yellow', 'light_blue', 'light_magenta', 'light_cyan', 'white']
    for i in range(pred_token_idx.size(0)):
        spec_stream(pred_token_idx[i], tokenizer, color_list[i])
    print()

def log_csv(file_path, header, entry):
    try:
        with open(file_path, 'r') as f:
            contents = f.read()
    except FileNotFoundError:
        contents = ""

    if not contents:
        with open(file_path, 'a') as f:
            f.write(header)
    
    with open(file_path, 'a') as f:
        f.write(entry)

def print_config(draft, target, prefill, gen_len, gamma, top_k, top_p, temperature, file_path, method, spec_args=None, dataset=None):
    print(colored("####################################### Config #######################################", 'blue'), flush=True)
    print(colored(f"Method: {method}", 'red'), flush=True)
    print(colored(f"Dataset: {dataset}", 'blue'), flush=True)
    print(colored(f"Spec Args: {spec_args}", 'blue'), flush=True)
    print(colored(f"Draft: {draft.config._name_or_path}", 'blue'), flush=True)
    print(colored(f"Target: {target.config._name_or_path}", 'blue'), flush=True)
    print(colored(f"Prefill Length: {prefill}", 'blue'), flush=True)
    print(colored(f"Generation Length: {gen_len}", 'blue'), flush=True)
    print(colored(f"Gamma: {gamma}", 'blue'), flush=True)
    print(colored(f"Sampling Method: top_k = {top_k}, top_p = {top_p}, temperature = {temperature}", 'blue'), flush=True)
    print(colored(f"Log CSV: {file_path}", 'blue'), flush=True)
    print(colored("######################################################################################\n", 'blue'), flush=True)

