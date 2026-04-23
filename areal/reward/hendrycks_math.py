from areal.utils import logging

from . import get_math_verify_worker

logger = logging.getLogger("HendrycksMathReward")


def hendrycks_math_reward_fn(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
) -> float:
    try:
        worker = get_math_verify_worker()
        return worker.verify(str(completions), str(answer))
    except Exception:
        logger.warning("Exception in hendrycks_math_reward_fn", exc_info=True)
        return 0.0
