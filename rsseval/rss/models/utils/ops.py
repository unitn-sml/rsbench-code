import torch


def outer_product(*tensors):
    # Check if the number of tensors is at least 2
    if len(tensors) < 2:
        raise ValueError(
            "At least two tensors are required for outer product calculation."
        )

    # Check if all tensors have the same shape
    shape = tensors[0].shape
    if any(tensor.shape != shape for tensor in tensors):
        raise ValueError("All input tensors must have the same shape.")

    # Create the einsum string dynamically based on the number of tensors
    einsum_string = ",".join(f"z{chr(97 + i)}" for i in range(len(tensors)))

    # Calculate the outer product
    result = torch.einsum(
        einsum_string + "->z" + "".join(chr(97 + i) for i in range(len(tensors))),
        *tensors,
    )

    return result
