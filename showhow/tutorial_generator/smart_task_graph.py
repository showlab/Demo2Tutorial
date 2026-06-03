import os
import json
import logging
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from openai import OpenAI

from .openai_compat import chat_completion_create


def _get_openai_config():
    return {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0")),
    }


openai_config = _get_openai_config()
openai_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global openai_client, openai_config
    if openai_client is None:
        openai_config = _get_openai_config()
        openai_client = OpenAI(api_key=openai_config["api_key"])
    return openai_client


logger = logging.getLogger(__name__)


def save_chunk_result(
    chunk_id: str, chunk_result: Dict[str, Any], output_dir: str
) -> None:
    """Save a chunk's processing result to a file."""
    output_path = Path(output_dir) / "chunks"
    output_path.mkdir(parents=True, exist_ok=True)
    chunk_file = output_path / f"{chunk_id}_result.json"
    with open(chunk_file, "w", encoding="utf-8") as f:
        json.dump(chunk_result, f, indent=2)
    logger.info(f"Saved chunk result to {chunk_file}")


def load_chunk_result(chunk_id: str, output_dir: str) -> Optional[Dict[str, Any]]:
    """Load a chunk's processing result from file if it exists."""
    chunk_file = Path(output_dir) / "chunks" / f"{chunk_id}_result.json"
    if chunk_file.exists():
        with open(chunk_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# --- Main Function ---
def generate_task_graph(
    all_actions: List[Dict[str, Any]],
    llm_prompt_template_path: str,
    max_actions_for_full_processing: int,
    overlap_size: int,
    max_chunk_size_for_llm: int,
    llm_merge_prompt_template_path: str = None,
    use_llm_merge: bool = True,
    max_steps_for_break: int = 10,
    output_dir: str = "output",
    critic_feedback: str = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a task graph from a sequence of GUI actions using smart chunking and context-aware merging.
    """
    if len(all_actions) <= max_actions_for_full_processing:
        logger.info("Processing full sequence with LLM.")
        prompt_template = Path(llm_prompt_template_path).read_text(encoding="utf-8")
        prompt_for_llm = format_prompt(prompt_template, all_actions, critic_feedback)
        task_graph_json = call_llm(prompt_for_llm, model=model)
        return task_graph_json
    else:
        logger.info("Sequence too long. Applying smart chunking and merging strategy.")
        list_of_chunks = smart_chunk_actions(
            all_actions, max_chunk_size_for_llm, overlap_size, max_steps_for_break
        )

        processed_chunk_graphs = []
        context_from_previous_chunk = None
        prompt_template = Path(llm_prompt_template_path).read_text(encoding="utf-8")

        for idx, chunk in enumerate(list_of_chunks):
            chunk_id = chunk["id"]
            logger.info(f"Processing chunk: {chunk_id}")

            # Try to load existing chunk result
            existing_result = load_chunk_result(chunk_id, output_dir)
            if existing_result is not None:
                logger.info(
                    f"Found existing result for chunk {chunk_id}, using cached result"
                )
                processed_chunk_graphs.append(existing_result)
                continue

            # Process new chunk
            prompt_for_chunk = format_prompt_for_chunk(
                prompt_template,
                chunk["actions"],
                context_from_previous_chunk,
                critic_feedback=critic_feedback,
            )
            try:
                current_chunk_graph_json = call_llm(prompt_for_chunk, model=model)
                # Save chunk result
                save_chunk_result(chunk_id, current_chunk_graph_json, output_dir)
                processed_chunk_graphs.append(current_chunk_graph_json)
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_id}: {str(e)}")
                # Save partial results
                if processed_chunk_graphs:
                    partial_result = {
                        "nodes": [],
                        "edges": [],
                        "error": f"Processing failed at chunk {chunk_id}: {str(e)}",
                        "processed_chunks": len(processed_chunk_graphs),
                    }
                    for g in processed_chunk_graphs:
                        partial_result["nodes"].extend(g.get("nodes", []))
                        partial_result["edges"].extend(g.get("edges", []))
                    return partial_result
                raise

        logger.info("Merging processed chunk graphs.")
        if use_llm_merge and llm_merge_prompt_template_path:
            merge_prompt_template = Path(llm_merge_prompt_template_path).read_text(
                encoding="utf-8"
            )
            final_merged_graph = merge_chunk_graphs_llm(
                processed_chunk_graphs, all_actions, merge_prompt_template, model=model
            )
        else:
            final_merged_graph = merge_chunk_graphs_heuristic(
                processed_chunk_graphs, all_actions
            )
        return final_merged_graph


def check_software_overlap(software_1, software_2):
    software_1 = str(software_1).lower()
    software_2 = str(software_2).lower()
    if software_1 in software_2 or software_2 in software_1:
        # print(f"Software overlap: {software_1} and {software_2}")
        return True
    else:
        return False


# --- Helper Functions ---
def smart_chunk_actions(
    all_actions: List[Dict[str, Any]],
    max_chunk_size: int,
    overlap: int,
    max_steps: int = 10,
) -> List[Dict[str, Any]]:
    """
    Smartly chunk actions, prioritizing software switches or long pauses, with overlap.
    Returns a list of chunk dicts: {id, actions, indices}

    Args:
        all_actions: List of action dictionaries
        max_chunk_size: Maximum number of actions in a chunk
        overlap: Number of actions to overlap between chunks
        max_steps: Maximum number of steps to look back when searching for a break point
    """
    chunks = []
    n = len(all_actions)
    start = 0
    chunk_id = 1
    while start < n:
        # Try to find a good break point within the next max_chunk_size
        end = min(start + max_chunk_size, n)
        # Prefer to break at software switch or long pause
        # import pdb; pdb.set_trace()
        if end < n:
            # Look back at most max_steps actions to find a break point
            look_back_start = max(start, end - max_steps)
            for i in range(end, look_back_start, -1):
                if i < n and (
                    not check_software_overlap(
                        all_actions[i]["software"], all_actions[i - 1]["software"]
                    )
                    or (
                        all_actions[i]["t_start"] - all_actions[i - 1]["t_end"] > 15
                    )  # 15s pause
                ):
                    end = i
                    break
        chunk_actions = all_actions[start:end]
        # Add overlap from previous chunk (except for the first chunk)
        if start != 0:
            overlap_start = max(0, start - overlap)
            chunk_actions = all_actions[overlap_start:end]
        chunks.append(
            {
                "id": f"chunk_{chunk_id}",
                "actions": chunk_actions,
                "indices": (start, end),
            }
        )
        start = end
        chunk_id += 1
    return chunks


def format_prompt(
    template: str, actions: List[Dict[str, Any]], critic_feedback: Optional[str] = None
) -> str:
    """
    Injects the actions into the LLM prompt template.
    """
    # Placeholder: you can customize this as needed
    # import pdb; pdb.set_trace()
    # user_message = template.replace("<INPUT_STEPS>", json.dumps(actions, indent=2))
    fb = f"\n\n[CRITIC_FEEDBACK]\n{critic_feedback}\n" if critic_feedback else ""
    return (
        template
        + fb
        + "\n\nHere is the actual input:\n"
        + json.dumps(actions, indent=2)
    )


def format_prompt_for_chunk(
    template: str,
    actions: List[Dict[str, Any]],
    previous_chunk_context: Optional[str],
    critic_feedback: str = None,
) -> str:
    fb = f"\n\n[CRITIC_FEEDBACK]\n{critic_feedback}\n" if critic_feedback else ""
    return (
        template
        + fb
        + "\n\nHere is the actual input:\n"
        + json.dumps(actions, indent=2)
    )


def call_llm(prompt_string: str, model: Optional[str] = None) -> Dict[str, Any]:
    """
    Sends the prompt to the LLM API and returns the parsed JSON graph.
    Expect STRICT JSON output. No prose.
    """
    selected_model = model or openai_config["model"]
    logger.debug("Calling LLM with model %s.", selected_model)
    resp = chat_completion_create(
        _get_client(),
        model=selected_model,
        temperature=openai_config["temperature"],
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a structured-output assistant. "
                    "Return ONLY valid JSON matching the requested schema. No explanations or prose."
                ),
            },
            {"role": "user", "content": prompt_string},
        ],
    )
    content = resp.choices[0].message.content

    # Print token usage for benchmark tracking
    if resp.usage:
        logger.info("Planner LLM: token usage %d", resp.usage.total_tokens)

    try:
        parsed = _parse_json_lenient(content)
    except json.JSONDecodeError:
        logger.warning("Planner LLM returned invalid JSON; requesting JSON repair.")
        parsed = _repair_json_with_llm(content, selected_model)

    logger.info("Successfully extracted structure")
    return parsed


def _parse_json_lenient(content: str) -> Dict[str, Any]:
    text = (content or "").strip()
    if not text:
        raise json.JSONDecodeError("empty response", content or "", 0)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fenced = re.search(
        r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL
    )
    if fenced:
        parsed = json.loads(fenced.group(1).strip())
        if isinstance(parsed, dict):
            return parsed

    candidate = _extract_first_json_object(text)
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("top-level JSON is not an object", candidate, 0)
    return parsed


def _extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise json.JSONDecodeError("no JSON object found", text, 0)

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    raise json.JSONDecodeError("unterminated JSON object", text, start)


def _repair_json_with_llm(content: str, model: str) -> Dict[str, Any]:
    repair_resp = chat_completion_create(
        _get_client(),
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "Convert the user's content into valid JSON only. Return no prose and no markdown.",
            },
            {
                "role": "user",
                "content": content,
            },
        ],
    )
    repaired = repair_resp.choices[0].message.content or ""
    try:
        return _parse_json_lenient(repaired)
    except json.JSONDecodeError as exc:
        logger.error("Planner LLM returned invalid JSON after repair.")
        logger.error("Original response: %s", content)
        logger.error("Repair response: %s", repaired)
        with open("llm_response.json", "w", encoding="utf-8") as f:
            f.write(content)
        with open("llm_response_repair.json", "w", encoding="utf-8") as f:
            f.write(repaired)
        raise ValueError(
            f"LLM returned invalid JSON after repair:\n{repaired}"
        ) from exc


def extract_context_for_next_chunk(graph_json: Dict[str, Any]) -> str:
    """
    Extracts a summary/context from a chunk's graph for the next chunk.
    """
    # Placeholder: could extract last node descriptions, etc.
    return "[context summary placeholder]"


def merge_chunk_graphs_llm(
    list_of_chunk_graphs: List[Dict[str, Any]],
    original_actions: List[Dict[str, Any]],
    merge_prompt_template: str,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Merge chunk graphs using LLM (placeholder).
    """
    # Call llm using the merge prompt template
    prompt_for_llm = format_prompt(merge_prompt_template, list_of_chunk_graphs)
    merged_graph_json = call_llm(prompt_for_llm, model=model)
    return merged_graph_json


def merge_chunk_graphs_heuristic(
    list_of_chunk_graphs: List[Dict[str, Any]], original_actions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Heuristic rule-based merging of chunk graphs.
    Handles node and edge merging while maintaining relationships and removing duplicates.
    """
    logger.info("[HEURISTIC MERGE] Merging chunk graphs...")

    # Initialize merged graph
    merged = {
        "nodes": [],
        "hierarchy_edges": [],
        "temporal_edges": [],
        "software_session_context": "Windows",  # Default value
        "graph_summary": "Merged task graph from multiple chunks",
    }

    # Track unique nodes by their ID to avoid duplicates
    unique_nodes = {}
    # Track unique edges by their source-target pair to avoid duplicates
    unique_hierarchy_edges = set()
    unique_temporal_edges = set()

    # Process each chunk graph
    for chunk_idx, chunk_graph in enumerate(list_of_chunk_graphs):
        # Process nodes
        for node in chunk_graph.get("nodes", []):
            node_id = node.get("id")
            if node_id not in unique_nodes:
                unique_nodes[node_id] = node
            else:
                # Merge node properties if they exist in both
                existing_node = unique_nodes[node_id]
                for key, value in node.items():
                    if key not in existing_node:
                        existing_node[key] = value
                    elif isinstance(value, list) and isinstance(
                        existing_node[key], list
                    ):
                        # Merge lists while removing duplicates
                        existing_node[key] = list(set(existing_node[key] + value))

        # Process hierarchy edges
        for edge in chunk_graph.get("hierarchy_edges", []):
            parent = edge.get("parent")
            child = edge.get("child")
            edge_id = f"{parent}->{child}:hierarchy"
            if edge_id not in unique_hierarchy_edges:
                unique_hierarchy_edges.add(edge_id)
                merged["hierarchy_edges"].append(edge)

        # Process temporal edges
        for edge in chunk_graph.get("temporal_edges", []):
            before = edge.get("before")
            after = edge.get("after")
            edge_id = f"{before}->{after}:temporal"
            if edge_id not in unique_temporal_edges:
                unique_temporal_edges.add(edge_id)
                merged["temporal_edges"].append(edge)

    # Add all unique nodes to merged graph
    merged["nodes"] = list(unique_nodes.values())

    # Add software context if available in any chunk
    for chunk_graph in list_of_chunk_graphs:
        if "software_session_context" in chunk_graph:
            merged["software_session_context"] = chunk_graph["software_session_context"]
            break

    # Add graph summary if available in any chunk
    summaries = []
    for chunk_graph in list_of_chunk_graphs:
        if "graph_summary" in chunk_graph:
            summaries.append(chunk_graph["graph_summary"])
    if summaries:
        merged["graph_summary"] = " | ".join(summaries)

    logger.info(
        f"[HEURISTIC MERGE] Merged {len(merged['nodes'])} nodes, "
        f"{len(merged['hierarchy_edges'])} hierarchy edges, and "
        f"{len(merged['temporal_edges'])} temporal edges"
    )
    return merged
