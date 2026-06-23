"""Convert .bpt -> .bptx (aligned, mmap-friendly) and save raw test sets."""
import os
import struct
import numpy as np
import torch


MAGIC = b"BPTX"
VERSION = 1
META_SLOT = 256   # bytes per tensor metadata slot
MAX_DIMS = 8
ALIGN = 64        # cache-line alignment for each data blob


def align_up(x, a):
    return (x + a - 1) & ~(a - 1)


def convert_bpt_to_bptx(src_path, dst_path):
    """Repack a .bpt file into a fixed-layout, 64-byte-aligned .bptx file
    suitable for memory-mapped / embedded inference runtimes."""
    tensors = []
    with open(src_path, "rb") as f:
        num = struct.unpack("I", f.read(4))[0]
        for _ in range(num):
            name_len = struct.unpack("I", f.read(4))[0]
            name = f.read(name_len).decode("utf-8")

            shape_len = struct.unpack("I", f.read(4))[0]
            shape = (struct.unpack(f"{shape_len}I", f.read(4 * shape_len))
                     if shape_len else ())

            is_packed = struct.unpack("B", f.read(1))[0]
            dtype_id = struct.unpack("B", f.read(1))[0]
            data_len = struct.unpack("I", f.read(4))[0]
            raw = f.read(data_len)

            tensors.append(dict(
                name=name, shape=shape, is_packed=bool(is_packed),
                dtype_id=dtype_id, raw=raw,
            ))

    header_size = 16
    meta_size = len(tensors) * META_SLOT
    data_start = align_up(header_size + meta_size, ALIGN)

    offsets = []
    cursor = data_start
    for t in tensors:
        offsets.append(cursor)
        cursor = align_up(cursor + len(t["raw"]), ALIGN)
    total_size = cursor

    with open(dst_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("I", VERSION))
        f.write(struct.pack("I", len(tensors)))
        f.write(struct.pack("I", data_start))

        for t, offset in zip(tensors, offsets):
            slot_start = f.tell()

            name_b = t["name"].encode("utf-8")[:127]
            f.write(name_b.ljust(128, b"\x00"))

            shape = t["shape"]
            numel = int(np.prod(shape)) if shape else 1
            f.write(struct.pack("I", len(shape)))
            padded_shape = list(shape) + [0] * (MAX_DIMS - len(shape))
            f.write(struct.pack(f"{MAX_DIMS}I", *padded_shape))

            f.write(struct.pack("B", int(t["is_packed"])))
            f.write(struct.pack("B", t["dtype_id"]))
            f.write(struct.pack("H", 0))

            f.write(struct.pack("Q", offset))
            f.write(struct.pack("Q", len(t["raw"])))

            row_bytes = (((shape[1] + 7) // 8)
                         if (t["is_packed"] and len(shape) >= 2) else 0)
            f.write(struct.pack("I", row_bytes))
            f.write(struct.pack("Q", numel))

            written = f.tell() - slot_start
            f.write(b"\x00" * (META_SLOT - written))

        for t, offset in zip(tensors, offsets):
            f.seek(offset)
            f.write(t["raw"])

        f.seek(total_size - 1)
        f.write(b"\x00")

    src_kb = os.path.getsize(src_path) / 1024
    dst_kb = os.path.getsize(dst_path) / 1024
    print(f"Converted {src_path} -> {dst_path}")
    print(f"  .bpt  {src_kb:.1f} KB")
    print(f"  .bptx {dst_kb:.1f} KB  "
          f"(overhead: {dst_kb - src_kb:.1f} KB metadata + alignment padding)")
    print(f"  Tensors: {len(tensors)}")
    for t, off in zip(tensors, offsets):
        print(f"    [{t['name']:30s}]  packed={t['is_packed']}  "
              f"shape={t['shape']}  offset=0x{off:08x}  bytes={len(t['raw'])}")


def save_testset(x, labels, path):
    """Save a test set as a flat binary blob (magic 'BTST').

    Layout: magic(4) | N(uint32) | L(uint32) | x float32[N,L] | labels int64[N].
    """
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
    else:
        x_np = np.ascontiguousarray(x, dtype=np.float32)

    if isinstance(labels, torch.Tensor):
        lbl_np = (labels.detach().cpu().contiguous().numpy()
                  .astype(np.int64, copy=False))
    else:
        lbl_np = np.ascontiguousarray(labels, dtype=np.int64)

    N, L = x_np.shape
    if lbl_np.shape[0] != N:
        raise ValueError(
            f"Sample count mismatch: x has {N} samples, "
            f"labels has {lbl_np.shape[0]}"
        )

    with open(path, "wb") as f:
        f.write(b"BTST")
        f.write(struct.pack("II", N, L))
        f.write(x_np.tobytes())
        f.write(lbl_np.tobytes())

    size_mb = (4 + 8 + N * L * 4 + N * 8) / 1024 ** 2
    print(f"Saved {N} samples (L={L}) -> {path} ({size_mb:.1f} MB)")
