"""DeepEP 2.5 EPBuffer adapter with the shared v2 dispatch/GEMM layout."""

from sglang.srt.layers.moe.token_dispatcher.deepep_v2 import (
    DeepEPv2Buffer,
    DeepEPv2Dispatcher,
    _DeepEPv2Impl,
)

_deepep_v2_5_import_error = None

try:
    from deep_ep import EPBuffer

    use_deepep_v2_5 = True
except (ImportError, OSError) as exc:
    use_deepep_v2_5 = False
    _deepep_v2_5_import_error = exc


class DeepEPv25Buffer(DeepEPv2Buffer):
    """Own an EPBuffer separately from the legacy v2 ElasticBuffer."""

    _STATE_KEY = "deepep_v2_5_ep_state"

    @staticmethod
    def _ensure_available() -> None:
        if not use_deepep_v2_5:
            detail = (
                f" Original import error: {_deepep_v2_5_import_error}"
                if _deepep_v2_5_import_error is not None
                else ""
            )
            raise ImportError(
                "--moe-a2a-backend deepep_v2.5 requires DeepEP 2.5 (EPBuffer). "
                "Install it from https://github.com/deepseek-ai/DeepEP." + detail
            )

    @staticmethod
    def _buffer_type():
        return EPBuffer


class _DeepEPv25Impl(_DeepEPv2Impl):
    buffer_class = DeepEPv25Buffer


class DeepEPv25Dispatcher(DeepEPv2Dispatcher):
    # EPBuffer's non-deferred dispatch/combine API uses the same output and
    # combine formats as v2, so the existing DeepGEMM adapters remain shared.
    impl_class = _DeepEPv25Impl
