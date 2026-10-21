"""Check interleaved environments, terminal rates, resets and sample intervals."""

import csv
import importlib.util
import json

import pytest

from conftest import PACKAGE_ROOT


@pytest.fixture
def rate_case(tmp_path):
    path = PACKAGE_ROOT / "scripts/process_clutter_ft_rates.py"
    spec = importlib.util.spec_from_file_location("_clutter_rates_cpu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source"
    source.mkdir()
    (source / "metadata.json").write_text(json.dumps({
        "status":"complete", "control_dt_s":.02, "physics_dt_s":.01, "csv_rows":8,
    }))
    rows = []
    for step in range(1,5):
        for env in range(2):
            reset = env == 0 and step >= 3
            row = dict(env_id=env, episode_id=int(reset), control_step=step,
                       physics_step=2*step, episode_step=step-2 if reset else step,
                       sim_time_s=step*.02, episode_time_s=(step-2 if reset else step)*.02,
                       sample_valid=1, delta_valid=int(step>1 and not (env==0 and step==3)),
                       terminated=int(env==0 and step==2), time_out=0,
                       tactile_count=env, goal_error_xy_m=.1)
            for frame in ("world","eef"):
                for axis in module.AXES:
                    unit="N" if axis.startswith("f") else "Nm"
                    row[f"v2_pure_{frame}_{axis}_{unit}"]=(1000*env + step**2)*(2 if frame=="eef" else 1)
            rows.append(row)

    def write():
        with (source / "wrench.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write()
    return module, source, tmp_path / "rates", rows, write


def test_axis_files_keep_terminal_rate_and_exclude_reset_jump(rate_case):
    module,source,output,_,_=rate_case
    report=module.process(source,output)
    assert report["input_rows"]==8 and report["output_rows_per_file"]==5
    assert report["excluded_invalid_delta_rows"]==3 and len(report["files"])==12
    for frame,multiplier in (("world",1),("eef",2)):
        for axis in module.AXES:
            unit="N" if axis.startswith("f") else "Nm"
            with (output/frame/f"{axis}_rate.csv").open() as f: rows=list(csv.DictReader(f))
            assert len(rows)==5
            assert [int(r["source_data_row"]) for r in rows]==[3,4,6,7,8]
            assert float(rows[0][f"rate_{unit}_per_s"])==pytest.approx(150*multiplier)
            assert int(rows[0]["terminated"])==1
            assert int(rows[3]["episode_id"])==1
            assert float(rows[3][f"rate_{unit}_per_s"])==pytest.approx(350*multiplier)


def test_reject_flag_that_would_create_derivative_across_reset(rate_case):
    module,source,output,rows,write=rate_case
    rows[4]["delta_valid"]=1;write()
    with pytest.raises(ValueError,match="boundary"): module.process(source,output)
    assert json.loads((output/"metadata.json").read_text())["status"]=="failed"


def test_reject_interval_inconsistent_with_physics_counter(rate_case):
    module,source,output,rows,write=rate_case
    rows[2]["physics_step"]=5;write()
    with pytest.raises(ValueError,match="interval"): module.process(source,output)
