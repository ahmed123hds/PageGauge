import json
from pathlib import Path

def score_prediction(task: str, prediction: str, references: list[str]) -> float:
    pred_str = str(prediction)
    pred_lower = pred_str.lower()
    
    if task == "niah_single_1":
        # Check if the reference needle (e.g. 7-digit number) is present
        target = references[0].strip()
        return 1.0 if target in pred_str else 0.0
        
    elif task == "niah_multikey_2":
        # Check if all reference keys/needles are present
        if not references:
            return 0.0
        matches = sum(1 for ref in references if ref.strip() in pred_str)
        return float(matches == len(references))
        
    elif task == "vt":
        # Variable tracking: fraction of reference variable names found
        if not references:
            return 0.0
        matches = sum(1 for ref in references if ref.strip() in pred_str)
        return float(matches / len(references))
        
    elif task == "cwe":
        # Common words extraction: fraction of reference words found
        if not references:
            return 0.0
        import re
        tokens = set(re.findall(r'\b[a-z]+\b', pred_lower))
        matches = sum(1 for ref in references if ref.lower().strip() in tokens or ref.lower().strip() in pred_lower)
        return float(matches / len(references))
        
    return 0.0

def test_against_base():
    base_dir = Path("/home/anonymous/pcaf_iclr27_e9_pretrained_ruler_v1/ruler_runs/base")
    for task_dir in base_dir.iterdir():
        if not task_dir.is_dir():
            continue
        pred_file = task_dir / "predictions.jsonl.partial"
        if not pred_file.exists():
            continue
        
        mismatches = 0
        total = 0
        with open(pred_file) as f:
            for line in f:
                d = json.loads(line)
                task = d["task"]
                pred = d["prediction"]
                refs = d["references"]
                expected = d["score"]
                computed = score_prediction(task, pred, refs)
                if abs(computed - expected) > 1e-4:
                    mismatches += 1
                total += 1
        print(f"Task {task_dir.name}: {total - mismatches}/{total} matched expected scores.")

if __name__ == "__main__":
    test_against_base()
