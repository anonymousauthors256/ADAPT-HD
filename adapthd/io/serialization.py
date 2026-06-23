"""Bit-packed serialization for bipolar HDC weights (.bpt format).

Selected parameters are quantized to {-1, +1}, packed to bits (8x storage
reduction), and stored with their original dtype so they can be restored.
"""
import struct
import numpy as np
import torch


DTYPE_TO_ID = {
    torch.float32: 0,
    torch.float16: 1,
    torch.float64: 2,
    torch.int64: 3,
    torch.int32: 4,
    torch.int16: 5,
    torch.int8: 6,
    torch.uint8: 7,
    torch.bool: 8,
}

ID_TO_NP_DTYPE = {
    0: np.float32,
    1: np.float16,
    2: np.float64,
    3: np.int64,
    4: np.int32,
    5: np.int16,
    6: np.int8,
    7: np.uint8,
    8: np.bool_,
}


def save_bipolar_model(model, packed_param_names, filename, bipolar_fn):
    """Serialize ``model.state_dict()`` to ``filename`` (.bpt).

    Parameters listed in ``packed_param_names`` are bipolarized via
    ``bipolar_fn(tensor, num_bits=1)`` and bit-packed; all others are stored
    as raw bytes.
    """
    state = model.state_dict()

    with open(filename, "wb") as f:
        f.write(struct.pack("I", len(state)))

        for name, tensor in state.items():
            tensor = tensor.detach().cpu().contiguous()

            name_bytes = name.encode("utf-8")
            f.write(struct.pack("I", len(name_bytes)))
            f.write(name_bytes)

            shape = list(tensor.shape)
            f.write(struct.pack("I", len(shape)))
            if len(shape) > 0:
                f.write(struct.pack(f"{len(shape)}I", *shape))

            if name in packed_param_names:
                f.write(struct.pack("B", 1))  # is_packed
                dtype_id = DTYPE_TO_ID[tensor.dtype]
                f.write(struct.pack("B", dtype_id))

                bipolar = bipolar_fn(tensor, num_bits=1)
                if not torch.all((bipolar == 1) | (bipolar == -1)):
                    raise ValueError(
                        f"bipolar_fn output for '{name}' contains values other "
                        f"than {{-1, +1}}. Unique values: "
                        f"{torch.unique(bipolar).tolist()}"
                    )

                binary = (bipolar > 0).to(torch.uint8)
                binary_np = binary.flatten().numpy()
                packed = np.packbits(binary_np)
                byte_data = packed.tobytes()

                f.write(struct.pack("I", len(byte_data)))
                f.write(byte_data)
            else:
                f.write(struct.pack("B", 0))  # is_packed
                dtype_id = DTYPE_TO_ID[tensor.dtype]
                f.write(struct.pack("B", dtype_id))

                byte_data = tensor.numpy().tobytes()
                f.write(struct.pack("I", len(byte_data)))
                f.write(byte_data)


def load_bipolar_model(model, filename, device="cpu"):
    """Load a .bpt file produced by :func:`save_bipolar_model` into ``model``."""
    loaded_state = {}

    with open(filename, "rb") as f:
        num_tensors = struct.unpack("I", f.read(4))[0]

        for _ in range(num_tensors):
            name_len = struct.unpack("I", f.read(4))[0]
            name = f.read(name_len).decode("utf-8")

            shape_len = struct.unpack("I", f.read(4))[0]
            if shape_len > 0:
                shape = struct.unpack(f"{shape_len}I", f.read(4 * shape_len))
            else:
                shape = ()

            is_packed = struct.unpack("B", f.read(1))[0]

            if is_packed:
                dtype_id = struct.unpack("B", f.read(1))[0]
                original_np_dtype = ID_TO_NP_DTYPE[dtype_id]

                data_len = struct.unpack("I", f.read(4))[0]
                raw = f.read(data_len)

                packed = np.frombuffer(raw, dtype=np.uint8)
                unpacked = np.unpackbits(packed)
                numel = int(np.prod(shape)) if len(shape) > 0 else 1
                unpacked = unpacked[:numel]
                if len(shape) > 0:
                    unpacked = unpacked.reshape(shape)

                arr = (unpacked.astype(np.float32) * 2.0 - 1.0).astype(
                    original_np_dtype
                )
                tensor = torch.from_numpy(arr.copy()).to(device)
            else:
                dtype_id = struct.unpack("B", f.read(1))[0]
                data_len = struct.unpack("I", f.read(4))[0]
                raw = f.read(data_len)

                arr = np.frombuffer(raw, dtype=ID_TO_NP_DTYPE[dtype_id]).copy()
                if len(shape) > 0:
                    arr = arr.reshape(shape)
                else:
                    arr = arr.reshape(())
                tensor = torch.from_numpy(arr).to(device)

            loaded_state[name] = tensor

    model.load_state_dict(loaded_state, strict=True)
    return model
