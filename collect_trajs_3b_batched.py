#!/usr/bin/env python
"""Collect 3B trajectories with batched generation, saving to /mnt/windows.
Verifies batching works correctly by using different prompts per batch item.
"""
import warnings; warnings.filterwarnings('ignore')
import gc, os, time, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = '/home/filip/Projects/qwen3b_awq'
OUT_DIR = '/mnt/windows/trajs_3b_batched'
N_LAYERS = 36
MAX_NEW = 120
N_TOTAL = 1000  # total trajectories to collect
BATCH_SIZES = [48]  # stable for 8GB GPU

def extract_trajs(plen, gen_len, hidden_states, n_layers):
    """Extract (h_seq, v_target) from batched hidden states.
    plen: prompt length (int), gen_len: (B,) per-item generated lengths.
    Returns list of (h, v) tuples, one per batch item.
    """
    B = len(gen_len)
    results = []
    for b in range(B):
        if gen_len[b] <= 1:
            results.append(None)
            continue
        hs_list = []
        for li in range(n_layers):
            h = hidden_states[li][b, plen:plen+gen_len[b]].float()
            hs_list.append(h)
        h_at = torch.stack(hs_list, dim=0)  # (L, T, D)
        h_seq = h_at[:, :-1]
        h_next = h_at[:, 1:]
        v = h_next - h_seq
        results.append((h_seq.permute(1, 0, 2).cpu().half(),
                        v.permute(1, 0, 2).cpu().half()))
    return results

def collect_batch(model, tok, problems, bs, n_layers, max_new):
    """Collect trajectories from problems at given batch size."""
    save_dir = os.path.join(OUT_DIR, f'bs{bs:02d}')
    os.makedirs(save_dir, exist_ok=True)

    buf_h, buf_v, batch_idx = [], [], 0
    count, t0 = 0, time.time()
    n_problems = min(len(problems), N_TOTAL)

    i = 0
    while count < n_problems:
        batch_probs = problems[i:i+bs]
        actual_bs = len(batch_probs)
        i += actual_bs

        prompts = [f"Q: {p['question']}\nA:" for p in batch_probs]
        inputs = tok(prompts, return_tensors='pt', padding=True).to('cuda')
        plen = inputs.input_ids.shape[1]

        # Generate
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        gen_lens = (out != tok.pad_token_id).sum(dim=1) - plen

        # Chunked second forward to avoid OOM on the logit tensor
        results = []
        chunk_size = 32
        for ch_start in range(0, actual_bs, chunk_size):
            ch_end = min(ch_start + chunk_size, actual_bs)
            out_chunk = out[ch_start:ch_end]
            gl_chunk = gen_lens[ch_start:ch_end]

            # Second forward for hidden states on this chunk
            with torch.no_grad():
                fwd_chunk = model(out_chunk, use_cache=False, output_hidden_states=True)
            chunk_results = extract_trajs(plen, gl_chunk, fwd_chunk.hidden_states, n_layers)
            results.extend(chunk_results)

            del out_chunk, fwd_chunk, chunk_results
            gc.collect()
            torch.cuda.empty_cache()

        del out, fwd, inputs
        gc.collect()
        torch.cuda.empty_cache()
        for r in results:
            if r is not None:
                buf_h.append(r[0]); buf_v.append(r[1])
                count += 1

        if len(buf_h) >= 50 or count >= n_problems:
            fname = os.path.join(save_dir, f'batch_{batch_idx:04d}.pt')
            torch.save({'hidden_seqs': torch.cat(buf_h).half(),
                        'velocity_targets': torch.cat(buf_v).half()}, fname)
            buf_h, buf_v = [], []; batch_idx += 1
            speed = count / (time.time() - t0) * 60
            print(f"  BS={bs:2d} [{count}/{n_problems}] batch_{batch_idx-1:04d}.pt ({speed:.0f} trajs/min)", flush=True)

    with open(os.path.join(save_dir, 'done.flag'), 'w') as f:
        f.write(f'n_trajectories={count}\n')
    print(f"  BS={bs:2d} DONE: {count} trajectories in {batch_idx} batches ({(time.time()-t0)/60:.0f} min)", flush=True)

def main():
    print(f"Loading {MODEL}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True,
                                                  device_map='cuda', torch_dtype=torch.float16)
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    tok.pad_token = tok.eos_token

    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = [r for r in ds if len(r["question"]) > 50]

    print(f"\nCollecting trajectories at various batch sizes...")
    print(f"Total: {N_TOTAL} trajectories per batch size")
    print(f"Max new tokens: {MAX_NEW}\n")

    for bs in BATCH_SIZES:
        collect_batch(model, tok, problems, bs, N_LAYERS, MAX_NEW)
        gc.collect()
        torch.cuda.empty_cache()

    # Verification
    print(f"\n{'='*60}")
    print(f"VERIFICATION: Checking trajectories are unique per batch item")
    print(f"{'='*60}")
    for bs in BATCH_SIZES:
        d = os.path.join(OUT_DIR, f'bs{bs:02d}')
        if os.path.exists(os.path.join(d, 'batch_0000.pt')):
            data = torch.load(os.path.join(d, 'batch_0000.pt'), map_location='cpu')
            h, v = data['hidden_seqs'], data['velocity_targets']
            # Check first two trajectories differ
            diff = (h[0] - h[1]).abs().mean().item()
            print(f"  BS={bs:2d}: {h.shape[0]} trajs, {h.shape} each. First-two diff={diff:.6f}", flush=True)

if __name__ == '__main__':
    main()
