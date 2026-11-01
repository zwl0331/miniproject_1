"""Process-isolated GPU queue with the existing raw-neural bank writer."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import traceback
from analysis.fast_exploration import write_json
from analysis.neural_exploration import run_trial, OUT
from songpop.phase2.raw_neural import Worker


def run(config):
    try:
        return run_trial(config)
    except Exception:
        error = {"id":config["id"],"status":"error","traceback":traceback.format_exc()}
        write_json(OUT / config["id"] / "error.json",error)
        print(error["traceback"],flush=True)
        Worker.close_all()
        return error


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--queue",required=True,type=Path)
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    queue=json.loads(args.queue.read_text())
    with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        results=list(pool.map(run,queue))
    write_json(args.queue.with_name(args.queue.stem+"_outcomes.json"),results)
    if any(r['status']=='error' for r in results):
        raise SystemExit(1)
