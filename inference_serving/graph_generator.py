import os
import shutil
import subprocess
import sys
from time import time
from .request import *
from .logger import get_logger

logger = get_logger("GraphGenerator")

GRAPH_ARTIFACT_STATS = {
    "workload_directories_generated": 0,
    "workload_directories_removed": 0,
    "trace_files_removed": 0,
}


def cleanup_batch_artifacts(hardware, model, instance_id, batch_id):
    """Remove dynamic artifacts once all ranks have consumed a batch."""
    file_name = f"{hardware}/{model}/instance{instance_id}_batch{batch_id}"
    trace_path = os.path.join("inputs", "trace", file_name + ".txt")
    workload_dir = os.path.join("inputs", "workload", file_name)
    if os.path.exists(trace_path):
        os.remove(trace_path)
        GRAPH_ARTIFACT_STATS["trace_files_removed"] += 1
    if os.path.isdir(workload_dir):
        shutil.rmtree(workload_dir)
        GRAPH_ARTIFACT_STATS["workload_directories_removed"] += 1


def get_graph_artifact_stats():
    return dict(GRAPH_ARTIFACT_STATS)


def generate_graph(batch, hardware, npu_num, node_id=0, instance_id=0, npu_offset=0, enable_local_offloading=False, event=False, in_memory=False):
    cwd = os.getcwd()
    chakra = os.path.join(cwd, "extern/graph_frontend/chakra")
    os.chdir(chakra)

    if event:
        file_name = 'event_handler'
    else:
        file_name = f'{hardware}/{batch.model}/instance{instance_id}_batch{batch.batch_id}'

    trace_path = f'../../../inputs/trace/{file_name}.txt'
    try:
        if in_memory:
            graph_frontend = os.path.dirname(chakra)
            if graph_frontend not in sys.path:
                sys.path.insert(0, graph_frontend)
            from chakra.src.converter.llm_converter import LLMConverter

            converter = LLMConverter(
                trace_path,
                "in-memory",
                npu_num,
                npu_offset,
                enable_local_offloading,
            )
            return converter.convert_to_payloads()

        workload_dir = f'../../../inputs/workload/{file_name}'
        os.makedirs(workload_dir, exist_ok=True)
        if not event:
            GRAPH_ARTIFACT_STATS["workload_directories_generated"] += 1

        cmd = (
            f'python -m chakra.src.converter.converter LLM '
            f'--input {trace_path} '
            f'--output ../../../inputs/workload/{file_name}/llm '
            f'--num-npus {npu_num} '
            f'--npu-offset {npu_offset}'
        )
        if enable_local_offloading:
            cmd += ' --local-offloading'

        logger.debug(
            "Generating graph with command: %s", cmd,
            extra={"node_id": node_id, "instance_id": instance_id},
        )
        subprocess.run(cmd.split(), text=True, check=True)
        return None
    finally:
        os.chdir(cwd)
