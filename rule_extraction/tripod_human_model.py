import json
from pathlib import Path
from typing import List

from tripod_exec_types import HumanFrameParams, HumanKeyframesOutput

class HumanParamModelInterface:
    def predict_keyframes(self, size: float) -> HumanKeyframesOutput:
        raise NotImplementedError


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


class StubHumanParamModel(HumanParamModelInterface):
    def predict_keyframes(self, size: float) -> HumanKeyframesOutput:
        shape_start = HumanFrameParams(0.42, 1.10, 0.25, 0.20, 0.25, 0.25)
        preshape    = HumanFrameParams(0.30, 0.85, 0.55, 0.50, 0.55, 0.50)
        enclosure   = HumanFrameParams(0.18, 0.60, 0.90, 0.90, 0.95, 0.90)
        final       = HumanFrameParams(0.16, 0.55, 1.00, 1.00, 1.00, 1.00)
        return HumanKeyframesOutput(shape_start, preshape, enclosure, final, 20, 25, 30)


class SummaryJSONModel(HumanParamModelInterface):
    def __init__(self, summary_json: Path):
        self.summary_json = Path(summary_json)

    def _safe_get(self, d, *keys, default=None):
        x = d
        for k in keys:
            if not isinstance(x, dict) or k not in x:
                return default
            x = x[k]
        return x

    def predict_keyframes(self, size: float) -> HumanKeyframesOutput:
        if not self.summary_json.exists():
            return StubHumanParamModel().predict_keyframes(size)

        data = json.loads(self.summary_json.read_text(encoding="utf-8"))

        def frame(name, fb):
            A = self._safe_get(data, "keyframes", name, "A", default=fb.A)
            D = self._safe_get(data, "keyframes", name, "D", default=fb.D)
            F = self._safe_get(data, "keyframes", name, "F", default=fb.F)
            ft = self._safe_get(data, "keyframes", name, "f_thumb", default=fb.f_thumb)
            fi = self._safe_get(data, "keyframes", name, "f_index", default=fb.f_index)
            fm = self._safe_get(data, "keyframes", name, "f_middle", default=fb.f_middle)
            return HumanFrameParams(float(A), float(D), float(F), float(ft), float(fi), float(fm))

        fb = StubHumanParamModel().predict_keyframes(size)
        return HumanKeyframesOutput(
            shape_start=frame("shape_start", fb.shape_start),
            preshape=frame("preshape", fb.preshape),
            enclosure=frame("enclosure", fb.enclosure),
            final=frame("final", fb.final),
            T_start_pre=int(self._safe_get(data, "timing", "T_start_pre", default=20)),
            T_pre_enc=int(self._safe_get(data, "timing", "T_pre_enc", default=25)),
            T_enc_final=int(self._safe_get(data, "timing", "T_enc_final", default=30)),
        )


class EnsembleSummaryModel(HumanParamModelInterface):
    """
    融合多个 summary.json（v1-v10）
    """
    def __init__(self, summary_json_list: List[Path]):
        self.models = [SummaryJSONModel(p) for p in summary_json_list]

    def predict_keyframes(self, size: float) -> HumanKeyframesOutput:
        outs = [m.predict_keyframes(size) for m in self.models]

        def avg_frame(getter):
            vals = [getter(o) for o in outs]
            return _mean(vals)

        def pack(stage_name):
            return HumanFrameParams(
                A=avg_frame(lambda o: getattr(getattr(o, stage_name), "A")),
                D=avg_frame(lambda o: getattr(getattr(o, stage_name), "D")),
                F=avg_frame(lambda o: getattr(getattr(o, stage_name), "F")),
                f_thumb=avg_frame(lambda o: getattr(getattr(o, stage_name), "f_thumb")),
                f_index=avg_frame(lambda o: getattr(getattr(o, stage_name), "f_index")),
                f_middle=avg_frame(lambda o: getattr(getattr(o, stage_name), "f_middle")),
            )

        return HumanKeyframesOutput(
            shape_start=pack("shape_start"),
            preshape=pack("preshape"),
            enclosure=pack("enclosure"),
            final=pack("final"),
            T_start_pre=int(round(avg_frame(lambda o: o.T_start_pre))),
            T_pre_enc=int(round(avg_frame(lambda o: o.T_pre_enc))),
            T_enc_final=int(round(avg_frame(lambda o: o.T_enc_final))),
        )
