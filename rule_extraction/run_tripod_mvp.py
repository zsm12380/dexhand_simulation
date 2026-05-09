# rule_extraction/run_tripod_mvp.py
from pathlib import Path
from tripod_exec_config import DEFAULT_XML
from tripod_human_model import EnsembleSummaryModel, StubHumanParamModel
from tripod_closedloop_executor import execute_closed_loop

def main():
    xml_path = str(DEFAULT_XML)
    size = 30.0
    dt = 0.01

    summary_dir = Path(__file__).resolve().parent / "outputs"
    summary_list = [summary_dir / f"tripod_size_30_v{i}_summary.json" for i in range(1, 11)]

    if all(p.exists() for p in summary_list):
        model = EnsembleSummaryModel(summary_list)
        print("[INFO] using v1-v10 ensemble summary")
    else:
        model = StubHumanParamModel()
        print("[WARN] summary missing, fallback stub")

    hk = model.predict_keyframes(size=size)
    execute_closed_loop(xml_path=xml_path, hk=hk, dt=dt, realtime=True)

if __name__ == "__main__":
    main()
