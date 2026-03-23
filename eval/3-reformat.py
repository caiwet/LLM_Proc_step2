import json
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import os

INPUT_DIR = '/data/bwh-comppath-img2/MGH_CID/LLM_Proc/benchmark/qwen3-4b-instruct_bs8_lr3e-5/benchmark/ehrcon_all_notes.parquet'
OUTPUT_DIR = '/data/bwh-comppath-img2/MGH_CID/LLM_Proc/benchmark/qwen3-4b-instruct_bs8_lr3e-5/reformat'

# -------------------- unified value extractors --------------------

def _extract_observation_values(res: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "valueQuantity_value": None,
        "valueQuantity_unit": None,
        "valueString": None,
        # "valueBoolean": None,
    }

    vq = res.get("valueQuantity")
    if isinstance(vq, dict):
        out["valueQuantity_value"] = vq.get("value")
        out["valueQuantity_unit"] = vq.get("unit")

    vs = res.get("valueString")
    if isinstance(vs, str):
        out["valueString"] = vs
    elif isinstance(vs, dict) and isinstance(vs.get("text"), str):
        out["valueString"] = vs.get("text")

    # vb = res.get("valueBoolean")
    # if isinstance(vb, bool):
    #     out["valueBoolean"] = vb

    return out

def _extract_medication_values(res: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ingredient_text": None,
        "ingredient_strength_numerator_value": None,
        "ingredient_strength_numerator_unit": None,
        "ingredient_strength_denominator_value": None,
        "ingredient_strength_denominator_unit": None,
        "dosageInstruction_text": None,
        "dosageInstruction_value": None,
        "dosageInstruction_unit": None,
        "timing_frequency": None,
        "timing_period": None,
        "timing_period_unit": None,
    }

    ing = res.get("ingredient")
    if isinstance(ing, dict):
        out["ingredient_text"] = ing.get("text")
        strength = ing.get("strength")
        if isinstance(strength, dict):
            num = strength.get("numerator")
            den = strength.get("denominator")
            if isinstance(num, dict):
                out["ingredient_strength_numerator_value"] = num.get("value")
                out["ingredient_strength_numerator_unit"] = num.get("unit")
            if isinstance(den, dict):
                out["ingredient_strength_denominator_value"] = den.get("value")
                out["ingredient_strength_denominator_unit"] = den.get("unit")

    dose = res.get("dosageInstruction")
    if isinstance(dose, dict):
        out["dosageInstruction_text"] = dose.get("text")
        quant = dose.get("quantity")
        if isinstance(quant, dict):
            out["dosageInstruction_value"] = quant.get("value")
            out["dosageInstruction_unit"] = quant.get("unit")

        timing = dose.get("timing")
        if isinstance(timing, dict):
            out["timing_frequency"] = timing.get("frequency")
            out["timing_period"] = timing.get("period")
            out["timing_period_unit"] = timing.get("periodUnit")

    return out

# -------------------- single flattener (no branching) --------------------

def flatten_resource(res: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    One stable schema for ALL resources.
    Anything not present stays null.
    """
    row: Dict[str, Any] = {
        "resource_index": idx,
        "resourceType": res.get("resourceType"),
        "id": res.get("id"),
        "status": res.get("status"),
        "description": res.get("description"),
        "body_site": res.get("bodySite"),
        "laterality": res.get("laterality"),
        "timestamp": res.get("timestamp"),
        "schedule_info": res.get("scheduleInfo"),

        # family-history-like (safe to always include)
        "relationship": res.get("relationship"),
    }

    # Always attempt both extractors (cheap; yields None when absent)
    row.update(_extract_observation_values(res))
    row.update(_extract_medication_values(res))

    return row

# -------------------- refs capture (unchanged) --------------------

def _split_ref(ref: str) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(ref, str):
        return None, None
    parts = ref.split("/")
    if len(parts) >= 2 and parts[0] and parts[1]:
        return parts[0], parts[1]
    return None, None

def walk_and_capture_refs(
    obj: Any,
    base_path: str,
    owner_type: Optional[str],
    owner_idx: int,
    owner_id: Optional[str]
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    if isinstance(obj, dict):
        if set(obj.keys()) == {"reference"} and isinstance(obj.get("reference"), str):
            target_ref = obj["reference"]
            target_type, target_id = _split_ref(target_ref)
            relation = "evidence" if ".evidence" in base_path else None
            out.append({
                "src_resource_index": owner_idx,
                "src_resourceType": owner_type,
                "src_id": owner_id,
                "path": base_path,
                "relation": relation,
                "target_ref": target_ref,
                "target_type": target_type,
                "target_id": target_id,
            })
        for k, v in obj.items():
            out += walk_and_capture_refs(v, f"{base_path}.{k}", owner_type, owner_idx, owner_id)

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            out += walk_and_capture_refs(item, f"{base_path}[{i}]", owner_type, owner_idx, owner_id)

    return out

# -------------------- public APIs (modified for JSON array) --------------------

def capture_codes_from_json_array(json_string: str) -> pd.DataFrame:
    """Process JSON array from a string."""
    rows: List[Dict[str, Any]] = []
    
    # Parse the JSON array
    resources = json.loads(json_string)
    
    # Ensure it's a list
    if not isinstance(resources, list):
        raise ValueError("Expected a JSON array")
    
    # Process each resource
    for r_idx, resource in enumerate(resources):
        if isinstance(resource, dict):
            rows.append(flatten_resource(resource, r_idx))
    
    return pd.DataFrame(rows)

def capture_refs_from_json_array(json_string: str) -> pd.DataFrame:
    """Process JSON array from a string."""
    edges: List[Dict[str, Any]] = []
    
    # Parse the JSON array
    resources = json.loads(json_string)
    
    # Ensure it's a list
    if not isinstance(resources, list):
        raise ValueError("Expected a JSON array")
    
    # Process each resource
    for r_idx, resource in enumerate(resources):
        if isinstance(resource, dict):
            rtype = resource.get("resourceType")
            rid = resource.get("id")
            edges += walk_and_capture_refs(resource, "$", rtype, r_idx, rid)
    
    return pd.DataFrame(edges)

# -------------------- CLI batch --------------------

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Read the parquet file
    print(f"Reading parquet file from: {INPUT_DIR}")
    df = pd.read_parquet(INPUT_DIR)

    df = df[df['valid_output'] == True].reset_index(drop=True)

    # df = df.sample(n=10, random_state=42).reset_index(drop=True)
    print(f"Found {len(df)} rows to process")


    
    # Process each row
    for idx, row in df.iterrows():
        try:
            row_id = row['ROW_ID']
            validated_output = row['concepts']
            
            print(f"\nProcessing row_id: {row_id} (row {idx+1}/{len(df)})")
            
            # Skip if validated_output is empty or null
            if pd.isna(validated_output) or not validated_output:
                print(f"  Skipping - empty validated_output")
                continue
            
            # Process codes
            flat_df = capture_codes_from_json_array(validated_output)
            flat_out = os.path.join(OUTPUT_DIR, f"{row_id}_codes.parquet")
            flat_df.to_parquet(flat_out, index=False)
            print(f"  Wrote codes: {flat_out} ({len(flat_df)} rows)")
            
            # Process refs
            refs_df = capture_refs_from_json_array(validated_output)
            refs_out = os.path.join(OUTPUT_DIR, f"{row_id}_refs.parquet")
            refs_df.to_parquet(refs_out, index=False)
            print(f"  Wrote refs: {refs_out} ({len(refs_df)} rows)")
            
        except Exception as e:
            print(f"  ERROR processing row_id {row_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
    
    print("\n✓ Processing complete!")