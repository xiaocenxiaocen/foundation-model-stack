import torch
import triton
from triton import language as tl
import sys
import marlin
import torch.nn as nn
from auto_gptq.utils.import_utils import dynamically_import_QuantLinear
from auto_gptq.modeling._utils import autogptq_post_init
import time
from a100_qlinear import a100_qlinear


@triton.jit()
def swizzle_tile(pid,
                m, n,
                block_m: tl.constexpr, block_n: tl.constexpr, group_m: tl.constexpr):

    grid_m = tl.cdiv(m, block_m)
    grid_n = tl.cdiv(n, block_n)

    width = group_m * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * group_m, group_m)

    pid_m = group_id * group_m + (pid % group_size)
    pid_n = (pid % width) // group_size

    return pid_m, pid_n


@triton.jit()
def matmul_data_parallel_kernel(a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr,
                             stride_am, stride_ak,
                             stride_bk, stride_bn,
                             stride_cm, stride_cn,
                             stride_scales_g, stride_scales_n,
                             stride_zeros_g, stride_zeros_n,
                             groupsize,
                             m, n, k,
                             block_size_m: tl.constexpr, block_size_n: tl.constexpr, block_size_k: tl.constexpr,
                             group_size_m: tl.constexpr,
                             fp8_fast_accum: tl.constexpr,):

    pid = tl.program_id(0)
    total_blocks_m = tl.cdiv(M, block_size_m)
    total_blocks_n = tl.cdiv(N, block_size_n)
    total_blocks_k = tl.cdiv(K, block_size_k)

    num_blocks_in_group = group_size_m * total_blocks_n
    group_id = pid // num_blocks_in_group
    group_size = min(total_blocks_m - group_id * group_size_m, group_size_m)

    pid_m = group_id * group_size_m + (pid % group_size)
    pid_n = (pid % num_blocks_in_group) // (group_size)

    offs_m = (pid_m * block_size_m + tl.arange(0, block_size_m)) % m
    offs_n = (pid_n * block_size_n + tl.arange(0, block_size_n)) % n

    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, block_size_m), block_size_m)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, block_size_n), block_size_n)
    offs_k = tl.arange(0, block_size_k)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak) # (16, 64)
    b_ptrs = b_ptr + ((offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn)

    scales_ptrs = scales_ptr + offs_bn * stride_scales_n
    zeros_ptrs = zeros_ptr + ((offs_bn // 8) * stride_zeros_n)

    shifter = (offs_k % 8) * 4
    zeros_shifter = (offs_bn % 8) * 4

    output = tl.zeros((block_size_m, block_size_n), dtype=tl.float32)
    for k in range(0, total_blocks_k):


        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

        # tl.device_print("data parallel b: ", b)

        g_id = k // (groupsize // block_size_k)

        ptr = scales_ptrs + g_id * stride_scales_g
        scales = tl.load(ptr)

        ptr = zeros_ptrs + g_id * stride_zeros_g
        zeros = tl.load(ptr)

        zeros = (zeros >> zeros_shifter) & 0xF
        zeros = (zeros + 1) * scales

        b = (b >> shifter[:, None]) & 0xF # b is int32
        b = b * scales[None, :] - zeros[None, :] # b is fp16

        # output +=  tl.dot(a, b)
        # output += tl.sum(a, b, axis=0)
        # print(b.type)
        # result = a[:, None] * b # (1 x 64 x 64 x 32) x illegal # (NEED A SQUARE MATRIX for B)
        # b -> 64 x 64 instead 64 x 32

        output += tl.dot(a, b)
        # a_block_ptr = tl.advance(a_block_ptr, (0, block_size_k))
        a_ptrs += stride_ak * block_size_k
        b_ptrs += (block_size_k//8) * stride_bk

    output.to(tl.float16)
    offs_cm = pid_m * block_size_m + tl.arange(0, block_size_m)
    offs_cn = pid_n * block_size_n + tl.arange(0, block_size_n)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    tl.store(c_ptrs, output)

class small_qlinear(torch.autograd.Function):
    def forward(ctx, a, b, scales, zeros):

        m, k = a.shape
        _, n = b.shape

        quant_groupsize = 128
        block_size_m = 16
        block_size_n = 32 # [N = 4096 // 32] = 128 blocks
        block_size_k = 128
        group_size_m = 8
        num_warps = 4
        num_stages = 3
        total_blocks_m = triton.cdiv(m, block_size_m)
        total_blocks_n = triton.cdiv(n, block_size_n)
        total_programs  = total_blocks_m * total_blocks_n
        grid = (total_programs, 1)
        fp8_fast_accum = False

        c = torch.zeros((m, n), device=b.device, dtype=torch.float16)
        # output = torch.em
        k = matmul_data_parallel_kernel[grid](
            a, b, c, scales, zeros,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            scales.stride(0), scales.stride(1),
            zeros.stride(0), zeros.stride(1),
            quant_groupsize,
            m, n, k,
            block_size_m, block_size_n, block_size_k, group_size_m, fp8_fast_accum = fp8_fast_accum,
            num_warps = num_warps, num_stages = num_stages,
        )

        #print(f"{k.n_regs} registers used, {k.n_spills} spills, {k.shared/1000} kB shared memory\n")

        #with open('dequant_simple.txt', 'w') as f:

        #    print(f"{k.n_regs} registers used, {k.n_spills} spills, {k.shared/1000} kB shared memory\n", file=f)
        #    print("IR", k.asm['ttir'], file=f)
        #    print("TTGIR", k.asm['ttgir'], file=f)
        #    print("PTX", k.asm['ptx'], file=f)
        #    print(f"{k.n_regs} registers used, {k.n_spills} spills, {k.shared/1000} kB shared memory\n", file=f)

        #    print(f"{total_blocks_m=} x {total_blocks_n=} = {total_programs=}")
        return c


matmul_data_parallel = small_qlinear.apply


@triton.jit()
def matmul_split_k_kernel(a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            stride_scales_g, stride_scales_n,
            stride_zeros_g, stride_zeros_n,
            groupsize,
            M, N, K,
            block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
            group_m: tl.constexpr, split_k: tl.constexpr):

    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_pid_k = tl.cdiv(K, block_k*split_k)

    pid_m, pid_n = swizzle_tile(pid,
                                M, N,
                                block_m, block_n, group_m)

    offs_m = pid_m*block_m + tl.arange(0, block_m)
    offs_n = pid_n*block_n + tl.arange(0, block_n)
    offs_k = pid_k*block_k + tl.arange(0, block_k)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_m, block_m), block_m)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, block_n), block_n)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + ((offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn)
    masks_am = offs_am < M

    scales_ptrs = scales_ptr + offs_bn * stride_scales_n
    zeros_ptrs = zeros_ptr + ((offs_bn // 8) * stride_zeros_n)

    shifter = (offs_k % 8) * 4
    zeros_shifter = (offs_bn % 8) * 4

    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k in range(0, num_pid_k):
        masks_k = offs_k < K
        masks_a = masks_am[:, None] & masks_k[None, :]
        a = tl.load(a_ptrs, mask=masks_a)
        b = tl.load(b_ptrs)

        g_id = (k * block_k * split_k) // groupsize

        ptr = scales_ptrs + g_id * stride_scales_g
        scales = tl.load(ptr) # -> 1D naive assumes no reordering

        ptr = zeros_ptrs + g_id * stride_zeros_g
        zeros = tl.load(ptr) # -> 1D naive assumes no reordering

        zeros = (zeros >> zeros_shifter) & 0xF
        zeros = (zeros + 1) * scales

        b = (b >> shifter[:, None]) & 0xF # b is int32
        b = b * scales[None, :] - zeros[None, :]

        acc += tl.dot(a, b)
        offs_k += block_k * split_k
        a_ptrs += block_k * split_k * stride_ak
        b_ptrs += (block_k//8) * split_k * stride_bk

    acc.to(tl.float16)

    offs_cm = pid_m*block_m + tl.arange(0, block_m)
    offs_cn = pid_n*block_n + tl.arange(0, block_n)

    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.atomic_add(c_ptrs, acc, mask=c_mask)

def matmul_split_k(a, b, scales, zeros):

    m, k = a.shape
    _, n = b.shape

    quant_groupsize = 128
    block_m = 16
    block_n = 32
    block_k = 128
    group_m = 8
    num_stages = 3
    num_warps = 4
    split_k = 4

    total_blocks_m = triton.cdiv(m, block_m)
    total_blocks_n = triton.cdiv(n, block_n)
    total_programs_mn = total_blocks_m * total_blocks_n
    total_programs_k = split_k

    grid = (total_programs_mn, total_programs_k)

    # print(f"problem m size: {m}, tile size m: {block_m}, total blocks m: {total_blocks_m}")
    # print(f"problem n size: {n}, tile size n: {block_n}, total blocks n: {total_blocks_n}")
    # print(f"problem k size: {k}, tile size k: {block_k}, total thread blocks k: {split_k}")
    # print(f"total thread blocks k: {k}, total thread blocks m and total thread blocks n = {total_blocks_m=} x {total_blocks_n} = {total_programs_mn}")


    # print(f"{total_programs_mn=}, {total_programs_k=}")

    c = torch.zeros((m, n), device=a.device, dtype=torch.float16)
    k = matmul_split_k_kernel[grid](a, b, c, scales, zeros,
                              a.stride(0), a.stride(1),
                              b.stride(0), b.stride(1),
                              c.stride(0), c.stride(1),
                              scales.stride(0), scales.stride(1),
                              zeros.stride(0), zeros.stride(1),
                              quant_groupsize,
                              m, n, k,
                              block_m, block_n, block_k,
                              group_m, split_k, num_stages=num_stages, num_warps=num_warps)

    # print(f"{k.n_regs} registers used, {k.n_spills} spills, {k.shared/1000} kB shared memory\n")

    # with open('matmul_split_k.txt', 'w') as f:

    #     print(f"{k.n_regs} registers used, {k.n_spills} spills, {k.shared/1000} kB shared memory\n", file=f)
    #     print("IR", k.asm['ttir'], file=f)
    #     print("TTGIR", k.asm['ttgir'], file=f)
    #     print("PTX", k.asm['ptx'], file=f)
    #     print(f"{k.n_regs} registers used, {k.n_spills} spills, {k.shared/1000} kB shared memory\n", file=f)

    return c

def make_tensor(M, N, dtype):
    if dtype == torch.int32:
        # Fill with random integers for int32 type
        res = torch.randint(low=-2**31, high=2**31, size=(M, N), dtype=dtype, device="cuda")
    else:
        # Fill with normally distributed random values for other types
        res = torch.empty((M, N), dtype=dtype, device="cuda")
        res.normal_(mean=0.0, std=0.5)
    return res


def gen_quant4(m, n, groupsize=-1):
    tile = 16
    maxq = 2 ** 4 - 1
    w = torch.randn((m, n), dtype=torch.half, device="cuda")
    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq
    w = torch.round(w / s).int()
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)
    ref = (w - (maxq + 1) // 2).half() * s
    if groupsize != -1:
        def reshape(w):
            w = w.reshape((groupsize, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((m, n)).contiguous()
            return w
        ref = reshape(ref)
        w = reshape(w)
    s = s.reshape((-1, n)).contiguous()
    linear = nn.Linear(m, n)
    linear.weight.data = ref.t()
    # Workaround to test some special cases that are forbidden by the API
    layer = marlin.Layer(256, 256, groupsize=groupsize)
    if groupsize == -1:
        groupsize = m
    layer.k = m
    layer.n = n
    layer.groupsize = groupsize
    layer.B = torch.empty((m // 16, n * 16 // 8), dtype=torch.int, device="cuda")
    layer.s = torch.empty((m // groupsize, n), dtype=torch.half, device="cuda")
    layer.pack(linear, s.t())
    q = layer.B
    s = layer.s
    return ref, q, s


def benchmark(f, warmup=10, iter=100):
    for i in range(warmup + iter):
        f()
        # We do not synchronize here in order to hide the kernel launch overhead during benchmarkining as this will also
        # happen during realistic model inference as many launches are submitted to the kernel queue.
        if i == warmup - 1:
            torch.cuda.synchronize()
            tick = time.time()
    torch.cuda.synchronize()
    res = (time.time() - tick) / iter
    # Make sure there is enough to "cool down" the GPU in between benchmarks to avoid throttling for later runs when
    # we execute many benchmarks consecutively
    time.sleep(1.)
    return res


def bench(f, warmup=5, number=1, repeat=31):
    import numpy as np

    cache = torch.empty((int(256e6),), dtype=torch.int8, device='cuda')
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(repeat)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(repeat)]

    for _ in range(warmup):
        cache.zero_()
        f()

    for i in range(repeat):
        cache.zero_()
        start_event[i].record()
        f()
        end_event[i].record()
    torch.cuda.synchronize()
    times = np.array([s.elapsed_time(e) for s, e in zip(start_event, end_event)])
    return np.mean(times).item()


def get_problem(m, n, k, groupsize=-1):
    if groupsize == -1:
        groupsize = k
    dev = torch.device('cuda:0')
    A = torch.randn((m, k), dtype=torch.half, device=dev)
    B = torch.randint(low=-3, high=3, size=(k//8, n), dtype=torch.int32, device=dev)
    B_ref = torch.randn((k, n), dtype=torch.half, device=dev)
    C = torch.zeros((m, n), dtype=torch.half, device=dev)
    s = torch.zeros((k // groupsize, n), dtype=torch.half, device=dev)
    z = torch.zeros((k // groupsize, n//8), dtype=torch.int32, device=dev)
    torch.cuda.synchronize()
    return A, B, C, B_ref, s, z


def benchmark_dense(A, B, C):
    res = bench(lambda: torch.matmul(A, B, out=C), repeat=100)
    return {
        's': res,
        'TFLOP/s': 2 * A.numel() * C.shape[1] / res / 10 ** 12,
        'GB/s': (2 * A.numel() + 2 * B.numel() + 2 * C.numel()) / res / 10 ** 9
    }


def benchmark_quant(A, B, C, s, z, thread_k, thread_n, sms, func=None):
    workspace = torch.zeros(C.shape[1] // 128 * 16, device=torch.device('cuda:0'))
    if func is None:
        def wrapper():
            marlin.mul(A, B, C, s, workspace, thread_k, thread_n, sms)
        f = wrapper
    else:
        def wrapper():
            func(A, B, s, z)
        f = wrapper
    res = bench(f, repeat=100)
    return {
        's': res,
        'TFLOP/s': 2 * A.numel() * C.shape[1] / res / 10 ** 12,
        'GB/s': (2 * A.numel() + 4 * B.numel() + 2 * C.numel() + 2 * s.numel()) / res / 10 ** 9
    }


def main():
    m = 8
    k = 4096 * 3
    n = 4096
    groupsize = 128
    g = k // groupsize

    a = make_tensor(m, k, dtype=torch.float16)
    b = make_tensor(k//8, n, dtype=torch.int32)
    c = make_tensor(m, n, dtype=torch.float16)
    workspace = torch.zeros(n//128*16, device="cuda")

    zeros = make_tensor(g, n//8, torch.int32)
    scales = make_tensor(g, n, torch.float16)


    # Marlin
    # m, n, k = 16, 4096, 4096
    # A = torch.randn((m, k), dtype=torch.half, device="cuda")
    # B_ref, B, s = gen_quant4(k, n)
    # C = torch.zeros((m, n), dtype=torch.half, device="cuda")
    # workspace = torch.zeros(n // 128*16, device="cuda")

    output_marlin = marlin.mul(a, b, c, scales, workspace, sms=108)
    output_split_k = matmul_split_k(a, b, scales, zeros)
    nbits = 4
    group_size=128
    disable_exllama=True
    disable_exllamav2=False
    use_triton = False

    linear_class = dynamically_import_QuantLinear(
    disable_exllama=disable_exllama, disable_exllamav2=disable_exllamav2,
    use_triton=use_triton, desc_act=False, group_size=group_size, bits=nbits)

    linear = linear_class(
    bits=nbits,
    group_size=group_size,
    infeatures=k,
    outfeatures=n,
    bias=0,
    )

    device = torch.device('cuda')

    linear.qweight = torch.randint(-100, 100, size=linear.qweight.shape, dtype=torch.int32)
    linear.scales = linear.scales + 0.002

    linear = linear.eval().to(device)
    linear = autogptq_post_init(linear, use_act_order=False)

    b_fake = torch.randn((k, n), dtype=torch.float16, device="cuda")

    # Warmup
    for i in range(3):
        linear(a)
        matmul_split_k(a, b, scales, zeros)
        torch.matmul(a, b_fake)


    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(s):
        matmul_split_k(a, b, scales, zeros)

    torch.cuda.current_stream().wait_stream(s)

    # capture
    g = torch.cuda.CUDAGraph()

    with torch.cuda.graph(g):
        matmul_split_k(a, b, scales, zeros)

    for i in range(7):
        torch.matmul(a, b_fake)

    for i in range(7):
        linear(a)

    for i in range(7):
        g.replay()  # This replays the captured operations in the graph


    for i in range(7):
        matmul_data_parallel(a, b, scales, zeros)

    for i in range(7):
        matmul_split_k(a, b, scales, zeros)


def main2():
    # Pass the SM count for known GPUs to avoid the kernel having to query this information (this is very minor)
    gpu = torch.cuda.get_device_name(0)
    if 'A100' in gpu:
        SMS = 108
    elif 'A10' in gpu:
        SMS = 72
    elif '3090' in gpu:
        SMS = 82
    elif 'A6000' in gpu:
        SMS = 84
    else:
        SMS = -1

    SMS = 128

    MODELS = {
        'Llama70B': [
            (8192, 8192),
            (8192, 28672),
            (28672, 8192),
        ],
        #'Llama7B': [
        #    (4096, 3 * 4096),
        #    (4096, 4096),
        #    (4096, 2 * 10752),
        #    (10752, 4096)
        #],
        #'Llama13B': [
        #    (5120, 3 * 5120),
        #    (5120, 5120),
        #    (5120, 2 * 13568),
        #    (13568, 5120)
        #],
        #'Llama33B': [
        #    (6656, 3 * 6656),
        #    (6656, 6656),
#       #     (6656, 2 * 17664),
        #    (17664, 6656)
        #],
        ##'Llama65B': [
        ##    (8192, 3 * 8192),
        ##    (8192, 8192),
        ##    (8192, 2 * 21760),
        ##    (21760, 8192)
        ##],
        #'Falcon180B': [
        #    # Note that parallel attention and FC allows layer fusions
#       #     (14848, 14848 * 5 + 1024),
        #    (14848 * 5, 14848)
        #]
    }

    # Set to true in order to run a more complete benchmark sweep; the default is reproduce README experiments
    ALL = False

    func = small_qlinear

    for groupsize in [-1, 128] if ALL else [128]:
        print('groupsize=%d' % groupsize)
        print()
        for model, layers in MODELS.items():
            print(model)
            if ALL:
                batchsizes =  [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
            else:
                batchsizes = [1, 8, 16, 32]
            for batch in batchsizes:
                for layer in layers:
                #if not ALL and model != 'ideal' and batch not in [1]:
                #    continue
                    A, B, C, B_ref, s, z = get_problem(batch, layer[1], layer[0], groupsize)
                    if model == 'ideal' and batch == 16:
                        # This is a special case constructed to be optimal for a thread-shape different than the default one
                        res_q = benchmark_quant(A, B, C, s, z, 64, 256, SMS, func=func)
                    else:
                        #try:
                        #    func = matmul_data_parallel
                        #    res_q = benchmark_quant(A, B, C, s, z, -1, -1, SMS, func=func)
                        #except RuntimeError:
                        #    pass
                        func = matmul_split_k
                        res_q = benchmark_quant(A, B, C, s, z, -1, -1, SMS, func=func)
                    print(f"{batch}x{layer[1]}x{layer[0]}:{res_q['s']}")
            print()


if __name__ == "__main__":
    main2()
