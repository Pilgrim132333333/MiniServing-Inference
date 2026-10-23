"""
方案B基准：真实 GPT2 prefill，端到端对比
Python for 循环 _write_to_cache vs CUDA 算子 reshape_and_cache
只替换写入实现，forward 其余路径完全一致，输出必须逐位一致
"""
import time
import statistics
import torch
from transformers import AutoModelForCausalLM

from miniserving.ModelRunner.GPT2ForCausalLM import GPT2ForCausalLM
from miniserving.kernels import _C

MODEL_PATH = "/root/models/gpt2"
BLOCK_SIZE = 16
NUM_BLOCKS = 64  # 共 1024 槽，覆盖最大 seq=512

class BenchConfig:
    num_blocks = NUM_BLOCKS
    block_size = BLOCK_SIZE

def make_cuda_write(engine):
    """把 _write_to_cache 换成 CUDA 算子版本，签名不变，forward 无感"""
    vec = engine.num_kv_heads * engine.head_dim

    def write(k_states, v_states, slot, layer_idx):
        k_cache = engine.k_tensor[layer_idx].view(-1, vec)
        v_cache = engine.v_tensor[layer_idx].view(-1, vec)
        # [B,H,S,D](非连续) -> [num_tokens, H*D]，reshape 触发一次 contiguous 拷贝
        k_vec = k_states.permute(0, 2, 1, 3).reshape(-1, vec)
        v_vec = v_states.permute(0, 2, 1, 3).reshape(-1, vec)
        slot_t = torch.as_tensor(slot, dtype=torch.int64, device=engine.device)
        _C.reshape_and_cache(k_vec, v_vec, k_cache, v_cache, slot_t)

    return write

def time_fn(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.mean(times), statistics.stdev(times)

class ModelWrapper:
    # 引擎 __init__ 约定: 传入对象.model == GPT2LMHeadModel
    def __init__(self, m):
        self.model = m


def main():
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    engine = GPT2ForCausalLM(ModelWrapper(model), BenchConfig())
    orig_write = engine._write_to_cache
    cuda_write = make_cuda_write(engine)

    print(f"\nGPT2-small | 12层 | prefill batch=1 | 每场景20次计时\n")
    print(f"{'seq_len':>8} | {'python循环(ms)':>16} | {'CUDA算子(ms)':>14} | {'加速比':>7} | {'logits':>6}")
    print("-" * 70)

    for seq_len in (64, 128, 256, 512):
        n_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        block_table = list(range(n_blocks))
        slot = list(range(seq_len))  # 连续分配时 flat index == block*16+offset
        inputs = [torch.randint(0, 50256, (seq_len,)).tolist()]
        start_pos = 0

        # 诊断第一步：单元级纯写入对比（不经过 forward，排除集成路径干扰）
        k_probe = torch.randn(1, engine.num_kv_heads, seq_len, engine.head_dim,
                              device="cuda", dtype=torch.float16)
        v_probe = torch.randn_like(k_probe)
        engine._write_to_cache = orig_write
        for p in engine.k_tensor: p.zero_()
        orig_write(k_probe, v_probe, slot, 0)
        k_py0 = engine.k_tensor[0].clone()
        for p in engine.k_tensor: p.zero_()
        cuda_write(k_probe, v_probe, slot, 0)
        torch.cuda.synchronize()
        if not torch.equal(engine.k_tensor[0], k_py0):
            d = (engine.k_tensor[0].float() - k_py0.float()).abs()
            s = int((d > 0).any(dim=tuple(range(1, d.dim()))).nonzero().flatten()[0])
            b, o = s // BLOCK_SIZE, s % BLOCK_SIZE
            print(f"  [unit] 写入不一致! slot={s} (block{b},off{o})")
            print(f"    python[0,:2,:4] = {k_py0[b, o, :2, :4].float().tolist()}")
            print(f"    cuda  [0,:2,:4] = {engine.k_tensor[0][b, o, :2, :4].float().tolist()}")
            print(f"    差异元素总数: {(d > 0).sum().item()} / {d.numel()}, 最大差: {d.max().item():.3e}")
            raise AssertionError("单元级写入对比失败 -> wrapper bug")
        print(f"  [unit seq={seq_len}] 单元级写入对比: 一致")

        # 诊断第二步：forward 集成对比
        engine._write_to_cache = orig_write
        for p in engine.k_tensor: p.zero_()
        for p in engine.v_tensor: p.zero_()
        logits_py = engine.forward(inputs, block_table, slot, start_pos)
        k_py = [t.clone() for t in engine.k_tensor]
        v_py = [t.clone() for t in engine.v_tensor]

        engine._write_to_cache = cuda_write
        for p in engine.k_tensor: p.zero_()
        for p in engine.v_tensor: p.zero_()
        logits_cu = engine.forward(inputs, block_table, slot, start_pos)
        torch.cuda.synchronize()

        k_same = all(torch.equal(a, b) for a, b in zip(engine.k_tensor, k_py))
        v_same = all(torch.equal(a, b) for a, b in zip(engine.v_tensor, v_py))
        out_diff = (logits_py.float() - logits_cu.float()).abs().max().item()
        print(f"  [diag seq={seq_len}] k_cache一致:{k_same} v_cache一致:{v_same} 输出最大差:{out_diff:.3e}")
        if not (k_same and v_same):
            for li, (a, b) in enumerate(zip(engine.k_tensor, k_py)):
                if not torch.equal(a, b):
                    bad = (a != b).any(dim=tuple(range(1, a.dim()))).nonzero().flatten()
                    raise AssertionError(f"层{li} K写入不一致! 差异slot(前10个): {bad[:10].tolist()}")
        assert out_diff == 0.0, f"seq={seq_len} 写入一致但输出不一致，疑似计算非确定性"
        engine._write_to_cache = orig_write

        mean_py, std_py = time_fn(lambda: engine.forward(inputs, block_table, slot, start_pos))
        engine._write_to_cache = cuda_write
        mean_cu, std_cu = time_fn(lambda: engine.forward(inputs, block_table, slot, start_pos))
        engine._write_to_cache = orig_write

        print(f"{seq_len:>8} | {mean_py:>9.2f}±{std_py:<5.2f} | {mean_cu:>7.2f}±{std_cu:<5.2f} | "
              f"{mean_py / mean_cu:>6.2f}x | {'PASS':>6}")

if __name__ == "__main__":
    main()