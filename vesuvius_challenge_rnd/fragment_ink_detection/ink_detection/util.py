import logging

import torch
import torch.nn as nn


def compile_if_possible(model: nn.Module) -> nn.Module:
    try:
        compiled_model = torch.compile(model)
        if not isinstance(compiled_model, nn.Module):
            logging.warning(f"Compiled model is not a nn.Module. Returning original model.")
            return model
        else:
            return compiled_model
    except Exception as e:
        logging.warning(f"Failed to compile model {model.__class__.__name__}: {e}.")
        return model
