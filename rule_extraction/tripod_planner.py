from tripod_exec_types import HumanKeyframesOutput, CtrlFrame, PlannedTrajectory
from tripod_mapper import map_human_to_synergy
from tripod_decoder import decode_u_to_ctrl

def _interp(a, b, t): return a + (b-a)*t

def _interp_ctrl(c1, c2, t):
    out = {}
    for k in set(c1.keys()) | set(c2.keys()):
        out[k] = _interp(c1.get(k, 0.0), c2.get(k, 0.0), t)
    return out

def build_trajectory(hk: HumanKeyframesOutput, dt=0.01) -> PlannedTrajectory:
    c1 = decode_u_to_ctrl(map_human_to_synergy(hk.shape_start))
    c2 = decode_u_to_ctrl(map_human_to_synergy(hk.preshape))
    c3 = decode_u_to_ctrl(map_human_to_synergy(hk.enclosure))
    c4 = decode_u_to_ctrl(map_human_to_synergy(hk.final))

    seq = []
    def add_seg(a: CtrlFrame, b: CtrlFrame, n: int):
        n = max(2, int(n))
        for i in range(n):
            t = i/(n-1)
            seq.append(CtrlFrame(ctrl=_interp_ctrl(a.ctrl, b.ctrl, t)))

    add_seg(c1, c2, hk.T_start_pre)
    add_seg(c2, c3, hk.T_pre_enc)
    add_seg(c3, c4, hk.T_enc_final)
    return PlannedTrajectory(ctrl_seq=seq, dt=dt)
