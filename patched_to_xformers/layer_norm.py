    except RuntimeError as e:
        # Catch cases where the current GPU does not have enough registers to hold a full tensor line
        # fallback to PyTorch's implementation, which streams the tensor in and out
        _triton_registered_warnings = True
        logger.warning(
            "Triton layernorm kernel register spillover or invalid image caught. "
            "Deactivating this kernel, please file an issue in the xFormers repository"
        )
        logger.warning(e)

        

  return torch.nn.functional.layer_norm(
        x,
        torch.Size([512]),
        weight=weight,
        bias=bias,
        eps=eps
    )