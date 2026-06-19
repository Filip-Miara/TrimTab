#!/home/filip/Projects/Personal/AI/Latent_Reasoning/qwen3_trm_env/bin/python -u
"""Exp 4: Train BottleneckTrajectoryTransformer on BnB data."""
import gc, json, os, sys, time, threading
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.adapters.trajectory_transformer import TrajectoryTransformer, BottleneckTrajectoryTransformer

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = "cuda"
DATA_DIR = "/run/media/filip/C27C20AB7C209C63/project_data/qwen25_7b_gen_trajs"
RMMAP_DIR = "/run/media/filip/C27C20AB7C209C63/trajs_7b_rmmap"
D_INPUT = 3584; N_LAYERS = 28; BS = 32; LR = 3e-4; N_EPOCHS = 20
CHUNK_SIZE = 2000; GPU_CHUNK = 2000

def compute_metrics(v_pred, v_target):
    v_p, v_t = v_pred.float(), v_target.float()
    mse = (v_p - v_t).pow(2).mean().item()
    var = v_t.var().item()
    r2 = 1.0 - mse / max(var, 1e-8)
    cos = (v_p * v_t).sum(-1) / (v_p.norm(dim=-1) * v_t.norm(dim=-1) + 1e-8)
    return r2, mse, cos.mean().item()

def build_mmap(files, val_idx):
    import glob
    os.makedirs(RMMAP_DIR, exist_ok=True)
    train_files = [f for i,f in enumerate(files) if i!=val_idx]
    val_file = files[val_idx]
    n_total = sum(len(torch.load(f,map_location="cpu")["hidden_seqs"]) for f in train_files)
    n_val = len(torch.load(val_file,map_location="cpu")["hidden_seqs"])
    print(f"Building 7B mmap ({n_total} train, {n_val} val)...",flush=True)
    h_map=np.memmap(f"{RMMAP_DIR}/train_h.bin",dtype="float16",mode="w+",shape=(n_total,N_LAYERS,D_INPUT))
    v_map=np.memmap(f"{RMMAP_DIR}/train_v.bin",dtype="float16",mode="w+",shape=(n_total,N_LAYERS,D_INPUT))
    val_h=np.memmap(f"{RMMAP_DIR}/val_h.bin",dtype="float16",mode="w+",shape=(n_val,N_LAYERS,D_INPUT))
    val_v=np.memmap(f"{RMMAP_DIR}/val_v.bin",dtype="float16",mode="w+",shape=(n_val,N_LAYERS,D_INPUT))
    offset=0
    for i,f in enumerate(train_files):
        d=torch.load(f,map_location="cpu"); n=len(d["hidden_seqs"])
        h_map[offset:offset+n]=d["hidden_seqs"].numpy(); v_map[offset:offset+n]=d["velocity_targets"].numpy()
        offset+=n; del d
        if (i+1)%10==0: print(f"  {i+1}/{len(train_files)}",flush=True)
    h_map.flush();v_map.flush()
    d=torch.load(val_file,map_location="cpu")
    val_h[:]=d["hidden_seqs"].numpy();val_v[:]=d["velocity_targets"].numpy()
    val_h.flush();val_v.flush();del d
    json.dump({"n_train":n_total,"n_val":n_val},open(f"{RMMAP_DIR}/meta.json","w"))
    print(f"mmap ready.",flush=True)

def load_chunk(h_mmap,v_mmap,ci,cs,nt,nl,di):
    s=ci*cs;e=min(s+cs,nt)
    h=torch.from_numpy(h_mmap[s:e].copy());v=torch.from_numpy(v_mmap[s:e].copy())
    cp=torch.randperm(len(h));return h[cp],v[cp]

def main():
    import glob, argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=N_EPOCHS)
    parser.add_argument("--bs",type=int,default=BS)
    parser.add_argument("--lr",type=float,default=LR)
    parser.add_argument("--rebuild",action="store_true")
    parser.add_argument("--resume",type=str,default=None)
    args=parser.parse_args()

    files=sorted(glob.glob(os.path.join(DATA_DIR,"gen_trajs_7b_batch_*.pt")))
    print(f"Total batch files: {len(files)}",flush=True)

    if args.rebuild or not os.path.exists(f"{RMMAP_DIR}/meta.json"):
        build_mmap(files,0)

    meta=json.load(open(f"{RMMAP_DIR}/meta.json"))
    n_train=meta["n_train"];n_val=meta["n_val"]
    print(f"Trajectories: {n_train} train, {n_val} val",flush=True)

    h_mmap=np.memmap(f"{RMMAP_DIR}/train_h.bin",dtype="float16",mode="r",shape=(n_train,N_LAYERS,D_INPUT))
    v_mmap=np.memmap(f"{RMMAP_DIR}/train_v.bin",dtype="float16",mode="r",shape=(n_train,N_LAYERS,D_INPUT))
    val_h_np=np.memmap(f"{RMMAP_DIR}/val_h.bin",dtype="float16",mode="r",shape=(n_val,N_LAYERS,D_INPUT))
    val_v_np=np.memmap(f"{RMMAP_DIR}/val_v.bin",dtype="float16",mode="r",shape=(n_val,N_LAYERS,D_INPUT))
    val_h=torch.from_numpy(val_h_np[:500].copy()).float()
    val_v=torch.from_numpy(val_v_np[:500].copy()).float()

    print("Computing stats...",flush=True)
    stat_v=torch.from_numpy(v_mmap[:10000].copy()).float()
    v_mean=stat_v.mean(dim=(0,1),keepdim=True)
    v_std=stat_v.std(dim=(0,1),keepdim=True)+1e-8
    v_mean_gpu=v_mean.to(DEVICE);v_std_gpu=v_std.to(DEVICE)
    val_v_norm=(val_v-v_mean)/v_std
    layer_w=torch.ones(N_LAYERS,device=DEVICE)/N_LAYERS

    # Bottleneck architecture: 1536→1280→1024→768→1024→1280→1536
    tt = BottleneckTrajectoryTransformer(
        dims=[1536, 1280, 1024, 768, 1024, 1280],
        n_heads=8, d_ff_ratio=4, n_positions=N_LAYERS, d_input=D_INPUT
    ).to(DEVICE)
    n_params=sum(p.numel() for p in tt.parameters())
    print(f"Bottleneck TT: {n_params:,} params",flush=True)

    if args.resume:
        tt.load_state_dict(torch.load(args.resume,map_location=DEVICE),strict=False)
        print(f"Resumed from {args.resume}",flush=True)

    try:
        tt = torch.compile(tt, mode="reduce-overhead", disable_cudagraphs=True)
        print(f"  torch.compile enabled",flush=True)
    except: pass

    opt=torch.optim.AdamW(tt.parameters(),lr=args.lr,weight_decay=1e-4)
    best_r2=-float("inf");t0=time.time()
    val_h_gpu=val_h.to(DEVICE);val_v_gpu=val_v.to(DEVICE)
    val_v_norm_gpu=val_v_norm.to(DEVICE)

    gpu_buf=[torch.empty(GPU_CHUNK,N_LAYERS,D_INPUT,dtype=torch.float16,device=DEVICE) for _ in range(2)]
    gpu_buf_v=[torch.empty(GPU_CHUNK,N_LAYERS,D_INPUT,dtype=torch.float16,device=DEVICE) for _ in range(2)]
    transfer_stream=torch.cuda.Stream()

    n_chunks=(n_train+CHUNK_SIZE-1)//CHUNK_SIZE
    print(f"Training: {n_chunks} chunks of {CHUNK_SIZE}, BS={args.bs}",flush=True)

    for epoch in range(args.epochs):
        tt.train();epoch_loss=0;n_batches=0
        chunk_order=np.random.permutation(n_chunks)
        first_h,first_v=load_chunk(h_mmap,v_mmap,chunk_order[0],CHUNK_SIZE,n_train,N_LAYERS,D_INPUT)
        n_cur=len(first_h)
        gpu_buf[0][:n_cur].copy_(first_h,non_blocking=True)
        gpu_buf_v[0][:n_cur].copy_(first_v,non_blocking=True)
        torch.cuda.synchronize();del first_h,first_v;active=0

        prefetch_ready=[False];prefetch_h=[None];prefetch_v=[None];n_nxt=[0]
        if n_chunks>1:
            def _pf():
                h,v=load_chunk(h_mmap,v_mmap,chunk_order[1],CHUNK_SIZE,n_train,N_LAYERS,D_INPUT)
                prefetch_h[0]=h;prefetch_v[0]=v;n_nxt[0]=len(h);prefetch_ready[0]=True
            threading.Thread(target=_pf,daemon=True).start()

        for ci in range(n_chunks):
            for bi in range(0,n_cur,args.bs):
                be=min(bi+args.bs,n_cur)
                hb=gpu_buf[active][bi:be].float()
                vb=gpu_buf_v[active][bi:be].float()
                vn=(vb-v_mean_gpu)/v_std_gpu
                loss=(layer_w*(tt(hb)-vn).pow(2).mean(dim=-1)).sum(dim=-1).mean()
                opt.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(tt.parameters(),1.0)
                opt.step();epoch_loss+=loss.item();n_batches+=1
            if ci+1<n_chunks:
                while not prefetch_ready[0]: time.sleep(0.001)
                staging=1-active
                with torch.cuda.stream(transfer_stream):
                    gpu_buf[staging][:n_nxt[0]].copy_(prefetch_h[0],non_blocking=True)
                    gpu_buf_v[staging][:n_nxt[0]].copy_(prefetch_v[0],non_blocking=True)
                torch.cuda.synchronize()
                if ci+2<n_chunks:
                    prefetch_ready[0]=False
                    def _pn(c=chunk_order[ci+2]):
                        h,v=load_chunk(h_mmap,v_mmap,c,CHUNK_SIZE,n_train,N_LAYERS,D_INPUT)
                        prefetch_h[0]=h;prefetch_v[0]=v;n_nxt[0]=len(h);prefetch_ready[0]=True
                    threading.Thread(target=_pn,daemon=True).start()
                active=staging;n_cur=n_nxt[0]
            print(f"  ep={epoch+1:2d} ch={ci+1}/{n_chunks} loss={epoch_loss/max(n_batches,1):.6f} {time.time()-t0:.0f}s",flush=True)
            gc.collect()

        tt.eval()
        with torch.no_grad():
            v_pred_norm=tt(val_h_gpu)
            v_pred=v_pred_norm*v_std_gpu+v_mean_gpu
            r2,mse,cos=compute_metrics(v_pred,val_v_gpu)
        if r2>best_r2:
            best_r2=r2
            model_to_save=tt._orig_mod if hasattr(tt,'_orig_mod') else tt
            torch.save(model_to_save.state_dict(),"best_tt_bottleneck.pt")
            json.dump({"d_model":"bottleneck","dims":[1536,1280,1024,768,1024,1280],"best_r2":r2},
                      open("best_tt_bottleneck.meta.json","w"))
            print(f"  ep={epoch+1:2d} loss={epoch_loss/max(n_batches,1):.6f} val_r²={r2:.4f} cos={cos:.4f} ✨ BEST {time.time()-t0:.0f}s",flush=True)
        else:
            print(f"  ep={epoch+1:2d} loss={epoch_loss/max(n_batches,1):.6f} val_r²={r2:.4f} cos={cos:.4f} {time.time()-t0:.0f}s",flush=True)

    print(f"\nBest val R²: {best_r2:.4f}",flush=True)

if __name__=="__main__":
    main()
