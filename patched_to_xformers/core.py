
    print("FFFALSe")
    if _has_cpp_library and mask.dtype == torch.bool and False:
        if isinstance(mask, SparseCS):
            return mask.matmul_with_mask(a, b)
        if mask.is_sparse:
            # perform broadcasting if needed
            mask = _broadcast_batch(mask, a.shape[0])

            # coalesced is not implemented for bool tensors, so need to cast
            mask = mask.to(dtype=a.dtype)  # type: ignore  # mypy is missing the catch above
        return torch.ops.xformers.matmul_with_mask(a, b, mask)