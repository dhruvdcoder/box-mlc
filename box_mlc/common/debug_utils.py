import torch
import logging

logger = logging.getLogger(__name__)


def analyse_tensor(
    t: torch.Tensor, debug_level: int = 1, name: str = ""
) -> None:
    if torch.isnan(t).any():
        if debug_level >= 2:
            breakpoint()
        else:
            logger.error(
                f"nan encountered in tensor {name} at indices {(t.isnan().nonzero())}"  # type:ignore
            )

    if torch.isinf(t).any():
        if debug_level >= 2:
            breakpoint()
        else:
            logger.error(
                f"inf encountered in tensor {name} at indices {(t.isinf().nonzero())}"  # type:ignore
            )
