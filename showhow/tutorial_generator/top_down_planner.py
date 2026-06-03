import json
import logging
from typing import List, Dict, Any
from pathlib import Path
import networkx as nx
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from .smart_task_graph import generate_task_graph

logger = logging.getLogger(__name__)


# --- Action dataclass and loader ---
@dataclass
class Action:
    step_idx: int
    software_name: str
    observation_before: str
    observation_after: str
    think: str
    action: str
    expectation: str
    start_time: float
    end_time: float
    action_trigger_timestamp: float
    debug_type: Optional[str] = None
    debug_points: Optional[List[Dict[str, float]]] = None


def load_trace(trace_file: str) -> List[Action]:
    try:
        with open(trace_file, "r", encoding="utf-8") as f:
            trace_data = json.load(f)
        actions = []
        for step in trace_data["trajectory"]:
            if "step_idx" not in step:
                raise KeyError(f"Missing 'step_idx' in trajectory entry: {step}")
            caption = step["caption"]
            time_info = step["time_info"]
            if "software_name" not in caption:
                caption["software_name"] = "Unknown"
            debug = step.get("caption_prompt_input[Debug]", {})  # 可能没有
            # 形如 [timestamp, "LClick at"/"Type: ..."/"LDrag", [{x,y}...]] 或 None
            dbg_action = debug.get("action")
            if isinstance(dbg_action, list) and len(dbg_action) >= 2:
                dbg_type = dbg_action[1]
                dbg_points = dbg_action[2] if len(dbg_action) > 2 else None
            else:
                dbg_type, dbg_points = None, None
            action = Action(
                step_idx=step["step_idx"],
                software_name=caption["software_name"],
                observation_before=caption["observation_action_before"],
                observation_after=caption["observation_action_after"],
                think=caption["think"],
                action=caption["action"],
                expectation=caption["expectation"],
                start_time=time_info["start_time"],
                end_time=time_info["end_time"],
                action_trigger_timestamp=time_info["current_action_trigger_timestamp"],
                debug_type=dbg_type,
                debug_points=dbg_points,
            )
            actions.append(action)
        return actions
    except Exception as e:
        logging.error(f"Failed to load trace: {e}")
        raise


def save_dag(
    dag: nx.DiGraph, software: str, graph_summary: str, output_dir: str, trace_id: str
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    graph_data = {
        "nodes": [{"id": node, **data} for node, data in dag.nodes(data=True)],
        "edges": [
            {"source": u, "target": v, "type": data["type"]}
            for u, v, data in dag.edges(data=True)
        ],
        "software": software,
        "graph_summary": graph_summary,
    }
    graph_file = output_path / f"{trace_id}_graph.json"
    with open(graph_file, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2)
    stats = {
        "total_nodes": len(dag.nodes),
        "total_edges": len(dag.edges),
        "hierarchy_edges": len(
            [e for e in dag.edges(data=True) if e[2]["type"] == "hierarchy"]
        ),
        "temporal_edges": len(
            [e for e in dag.edges(data=True) if e[2]["type"] == "temporal"]
        ),
        "software_apps": list(
            set(data.get("software", "") for _, data in dag.nodes(data=True))
        ),
        "generation_time": datetime.now().isoformat(),
    }
    stats_file = output_path / f"{trace_id}_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved graph and statistics to {output_dir}")


def dict_graph_to_nx(graph_json: Dict[str, Any]) -> nx.DiGraph:
    G = nx.DiGraph()
    # Add nodes
    for node in graph_json.get("nodes", []):
        node_id = node["id"]
        G.add_node(node_id, **node)

    # Add hierarchy edges
    for edge in graph_json.get("hierarchy_edges", []):
        G.add_edge(edge["parent"], edge["child"], type="hierarchy")

    # Add temporal edges
    for edge in graph_json.get("temporal_edges", []):
        G.add_edge(edge["before"], edge["after"], type="temporal")

    return G


def _next_iter_id(base_dir: Path) -> int:
    """Find next available iteration number."""
    existing = [p.stem for p in base_dir.glob("tut_draft_iter_*.json")]
    nums = []
    for name in existing:
        try:
            num = int(name.split("_")[-1])
            nums.append(num)
        except ValueError:
            pass
    return (max(nums) + 1) if nums else 1


def trace_to_tut_draft(
    trace_file: str,
    output_dir: str = "output",
    prompt_file: str = "prompt_top_down_planning.txt",
    max_actions_for_full_processing: int = 200,
    max_chunk_size_for_llm: int = 50,
    overlap_size: int = 5,
    merge_prompt_file: str = None,
    use_llm_merge: bool = False,
    max_steps_for_break: int = 10,
    critic_feedback: str = None,
    iter_id: int = None,
    model: str = None,
):
    actions = load_trace(trace_file)
    base_dir = Path(trace_file).parent / "tut_draft"
    base_dir.mkdir(parents=True, exist_ok=True)

    # 自动分配 iter_id
    if iter_id is None:
        iter_id = _next_iter_id(base_dir)

    out_path = base_dir / f"tut_draft_iter_{iter_id:03d}.json"

    # Select only the necessary fields for task graph generation
    actions_dicts = []
    for action in actions:
        action_dict = {
            "action_id": action.step_idx,
            "software": action.software_name,
            "state_before": action.observation_before,
            "state_after": action.observation_after,
            "action_desc": action.action,
            "t_start": action.start_time,
            "t_end": action.end_time,
            "raw_action": {
                "type": action.debug_type,  # e.g., "LClick at", "Type: 50", "LDrag"
                "points": action.debug_points,  # list of {x,y} or None
            },
        }
        actions_dicts.append(action_dict)

    logger.info(
        "Planner: generating draft for %d actions (iter %s)",
        len(actions_dicts),
        iter_id,
    )
    graph_json = generate_task_graph(
        all_actions=actions_dicts,
        llm_prompt_template_path=prompt_file,
        max_actions_for_full_processing=max_actions_for_full_processing,
        overlap_size=overlap_size,
        max_chunk_size_for_llm=max_chunk_size_for_llm,
        llm_merge_prompt_template_path=merge_prompt_file,
        use_llm_merge=use_llm_merge,
        max_steps_for_break=max_steps_for_break,
        output_dir=str(base_dir),
        critic_feedback=critic_feedback,
        model=model,
    )
    # dag = dict_graph_to_nx(graph_json)
    # try:
    #     software_session_context = graph_json["software_session_context"]
    # except:
    #     software_session_context = "Windows"
    # try:
    #     graph_summary = graph_json["graph_summary"]
    # except:
    #     graph_summary = "No summary available"
    # save_dag(dag, software_session_context, graph_summary, output_dir, trace_id)
    # return dag
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(graph_json, f, indent=2)
    logger.info("Planner: saved draft -> %s", out_path)
    return out_path
