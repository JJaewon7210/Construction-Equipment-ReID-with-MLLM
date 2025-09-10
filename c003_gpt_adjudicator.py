from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import os
import json
import base64
import tempfile
from pathlib import Path
from openai import OpenAI

# =========================
# OpenAI client
# =========================
_client = None
def _get_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client

# =========================
# Data classes
# =========================
@dataclass
class CandidateTrack:
    tid: int
    delta_t: float
    color_name: Optional[str] = None  # e.g., red/blue/green/orange/purple


@dataclass
class NewIdEvent:
    current_idx: int
    class_name: str
    frame_width: int
    frame_height: int
    fps: float
    new_id: str
    current_overlay_path: Optional[str] = None  # single NEW image (optional)
    candidates: List[CandidateTrack] = field(default_factory=list)
    participants: List[Dict[str, Any]] = field(default_factory=list)  # participants for the NEW image
    past_overlays: List[Dict[str, Any]] = field(default_factory=list)  # list of single-candidate overlay dicts

# =========================
# JSON schema for structured response
# =========================
ADJUDICATION_JSON_SCHEMA = {
    "name": "adjudication_schema",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "new_id": {"type": "string"},
            "new_id_decision": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["assign", "new", "delete"],
                    },
                    "target_id": {"type": ["integer", "null"]},
                },
                "required": ["action", "target_id"],
            },
        },
        "required": ["new_id", "new_id_decision"],
    },
}

# =========================
# System instruction
# =========================
SYS_INST = """\
<role>
You are a visual identity adjudicator. Your task is to decide whether the NEW object (a pink bounding box) is the same object as exactly one of the candidates shown in the candidate images (each with a colored bounding box), or whether it is a truly NEW object not present among the candidates.
</role>

<inputs>
1. A <scene> block with metadata about the video frame (expected class, frame size, fps).
2. A <new> block, which includes:
   - The class and bounding box color of the NEW object
   - An image showing the NEW object in context with a pink bounding box
3. Multiple <candidate> blocks, each containing:
   - An image of a single candidate object with a unique bounding box color (e.g., red/orange/green/blue/purple)
   - The candidate’s track_id and bbox_color are visually indicated in the image

</inputs>

<task>
Decide if the NEW (pink-box) object is the same identity as one of the candidates.
- If it matches a candidate:
  - new_id_decision.action = "assign"
  - new_id_decision.target_id = that candidate’s track_id (as an integer)
- If it does not match any candidate:
  - new_id_decision.action = "new"
  - new_id_decision.target_id = null
- If the NEW object is NOT of the expected class specified in <scene> (i.e., class_name mismatch):
  - new_id_decision.action = "delete"
  - new_id_decision.target_id = null
</task>
"""

# =========================
# Helpers
# =========================

def _to_data_url(path: str) -> str:
    """Encode a JPEG file as a data URL (base64)."""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _build_summary_text(event: NewIdEvent) -> str:
    """Build the minimal scene header for the user content."""
    lines = ["<scene>", f"class={event.class_name} frame={event.frame_width}x{event.frame_height} fps={event.fps:.2f}", "</scene>"]
    return "\n".join(lines)


def _append_candidate_bundle(user_blocks: List[Dict[str, Any]], candidate: CandidateTrack, overlay: Optional[Dict[str, Any]]) -> None:
    """Append a candidate text/image bundle to the user blocks."""
    user_blocks.append({"type": "text", "text": "<candidate>"})
    user_blocks.append({"type": "text", "text": "image="})
    user_blocks.append({
        "type": "image_url",
        "image_url": {"url": _to_data_url(overlay["path"]), "detail": "low"},
    })
    user_blocks.append({"type": "text", "text": "</candidate>"})


def _index_candidate_overlays_by_tid(event: NewIdEvent) -> Dict[int, Dict[str, Any]]:
    """Index single-candidate overlay dicts by their participant tid.

    Each overlay is expected to include a participants list where the first
    participant dict has role "candidate" and a numeric tid.
    """
    idx: Dict[int, Dict[str, Any]] = {}
    for overlay in event.past_overlays or []:
        parts = overlay.get("participants") or []
        if not parts:
            continue
        p0 = parts[0]
        if p0.get("role") != "candidate":
            continue
        tid = p0.get("tid")
        if tid is None:
            continue
        try:
            idx[int(tid)] = overlay
        except Exception:
            continue
    return idx

# =========================
# Main entry
# =========================

def adjudicate_new_id(
    event: NewIdEvent,
    model: str = "gpt-5-mini",
    max_past: int = 10
    ) -> Dict[str, Any]:
    """Run the adjudication request and return a compact dict response.

    The function prepares a minimal structured user payload that includes
    one NEW image (if available) and up to `max_past` single-candidate images.
    """
    client = _get_client()

    # 1) Header text block
    user_blocks: List[Dict[str, Any]] = [{"type": "text", "text": _build_summary_text(event)}]

    # 2) NEW image block (if present)
    new_items = [p for p in (event.participants or []) if p.get("role") == "new"]
    new_item = new_items[0] if new_items else {}
    new_class = new_item.get("class_name", event.class_name or "unknown")
    new_color = new_item.get("color", "pink box")

    if event.current_overlay_path:
        user_blocks.append({"type": "text", "text": "<new>"})
        user_blocks.append({"type": "text", "text": f"class={new_class} bbox_color={new_color}"})
        user_blocks.append({"type": "text", "text": "image="})
        user_blocks.append({"type": "image_url", "image_url": {"url": _to_data_url(event.current_overlay_path), "detail": "low"}})
        user_blocks.append({"type": "text", "text": "</new>"})

    # 3) Candidate single-overlay images
    overlay_by_tid = _index_candidate_overlays_by_tid(event)
    for candidate in event.candidates[:max_past]:
        overlay = overlay_by_tid.get(int(candidate.tid))
        _append_candidate_bundle(user_blocks, candidate, overlay)

    # 4) Call the model using structured output formatting
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYS_INST},
            {"role": "user", "content": user_blocks},
        ],
        response_format={"type": "json_schema", "json_schema": ADJUDICATION_JSON_SCHEMA},
        max_completion_tokens=8000,
        seed=42,
    )

    raw = completion.choices[0].message.content
    data = json.loads(raw)

    # 5) Post-process and ensure expected fields
    data.setdefault("new_id", str(event.new_id))
    nid = data.get("new_id_decision", {}) or {}
    data["new_id_decision"] = nid
    data.setdefault("existing_ids_review", [])
    if nid.get("action") == "delete": nid["target_id"] = None

    # 6) Save the full completion payload next to the overlay if possible
    out_root = Path(event.current_overlay_path).parent if event.current_overlay_path else Path.cwd()
    save_dir = out_root / "gpt_completion"
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(event.current_overlay_path).stem if event.current_overlay_path else "completion"
        gpt_completion_path = save_dir / f"{stem}.json"
        # model_dump() is used in the v1 OpenAI Python SDK
        gpt_completion = completion.model_dump()
        fd, tmp_path = tempfile.mkstemp(dir=str(save_dir), suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fp_out:
            json.dump(gpt_completion, fp_out, ensure_ascii=False, indent=4)
        os.replace(tmp_path, gpt_completion_path)
        data.setdefault("meta", {})["gpt_completion_path"] = str(gpt_completion_path)
    except Exception:
        # If saving fails, do not interrupt the main response
        pass

    return data
