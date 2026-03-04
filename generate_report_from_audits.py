"""
Generate report_2.csv files from audit_EvaluationX.json files.
Creates new ground truth reports in CSV format based on audit results.
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, List, Any


def get_status_from_score(score: Any) -> str:
    """
    Determine status based on score.
    
    Args:
        score: Score value (int, float, or str)
    
    Returns:
        Status string (PASS, PARTIAL, or FAIL)
    """
    try:
        if isinstance(score, str):
            if score.upper() == 'N/A':
                return 'N/A'
            score = float(score)
        
        if score >= 4:
            return 'PASS'
        elif score >= 2:
            return 'PARTIAL'
        else:
            return 'FAIL'
    except (ValueError, TypeError):
        return 'N/A'


def generate_report_from_audit(audit_json_path: str, output_csv_path: str) -> None:
    """
    Generate a report_2.csv from an audit JSON file.
    
    Args:
        audit_json_path: Path to audit_EvaluationX.json file
        output_csv_path: Path where report_2.csv will be saved
    """
    # Load audit JSON
    print(f"📖 Reading: {audit_json_path}")
    with open(audit_json_path, 'r', encoding='utf-8') as f:
        audit_data = json.load(f)
    
    # Handle both list format (direct array) and dict format (with 'requirements' key)
    if isinstance(audit_data, list):
        requirements = audit_data
    else:
        requirements = audit_data.get('requirements', [])
    print(f"   Found {len(requirements)} requirements")
    
    # Prepare CSV rows
    csv_rows = []
    for req in requirements:
        req_id = req.get('Requirement_ID', '')
        req_name = req.get('Requirement_Name', '')
        score = req.get('Score', 'N/A')
        auditor_notes = req.get('Auditor_Notes', '')
        
        # Derive status from score
        status = get_status_from_score(score)
        
        csv_rows.append({
            'Requirement_ID': req_id,
            'Requirement Name': req_name,
            'Status': status,
            'Score': score,
            'Auditor Notes': auditor_notes
        })
    
    # Write CSV
    print(f"💾 Writing: {output_csv_path}")
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['Requirement_ID', 'Requirement Name', 'Status', 'Score', 'Auditor Notes']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(csv_rows)
    
    print(f"✓ Generated report with {len(csv_rows)} rows")


def main():
    """Generate report_2.csv files for all audit_EvaluationX.json files."""
    
    print("=" * 80)
    print("Generate report_2.csv from Audit JSON Files")
    print("=" * 80)
    
    debug_dir = Path("data/debug")
    ground_truth_base = Path("data/ground_truth/raw_data")
    
    if not debug_dir.exists():
        print(f"❌ Debug directory not found: {debug_dir}")
        return
    
    # Find all audit_EvaluationX.json files
    audit_files = sorted(debug_dir.glob("audit_Evaluation*.json"))
    
    if not audit_files:
        print(f"⚠️  No audit_Evaluation*.json files found in {debug_dir}")
        return
    
    print(f"\n📁 Found {len(audit_files)} audit files:")
    for f in audit_files:
        print(f"   - {f.name}")
    
    print("\n" + "=" * 80)
    
    # Process each audit file
    success_count = 0
    for audit_file in audit_files:
        # Extract evaluation name (e.g., "Evaluation1" from "audit_Evaluation1.json")
        eval_name = audit_file.stem.replace("audit_", "")
        
        # Construct output path
        output_dir = ground_truth_base / eval_name
        output_path = output_dir / "report_2.csv"
        
        print(f"\n[{eval_name}]")
        
        if not output_dir.exists():
            print(f"⚠️  Output directory not found: {output_dir}")
            print(f"   Creating directory...")
            output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            generate_report_from_audit(str(audit_file), str(output_path))
            success_count += 1
            
            # Print summary
            file_size = output_path.stat().st_size
            print(f"✓ Success ({file_size:,} bytes)")
            
        except Exception as e:
            print(f"❌ Error processing {audit_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print("\n" + "=" * 80)
    print(f"✅ Completed: {success_count}/{len(audit_files)} reports generated")
    print("=" * 80)
    
    if success_count > 0:
        print("\nGenerated files:")
        for audit_file in audit_files:
            eval_name = audit_file.stem.replace("audit_", "")
            report_path = ground_truth_base / eval_name / "report_2.csv"
            if report_path.exists():
                size = report_path.stat().st_size
                print(f"  ✓ {report_path} ({size:,} bytes)")


if __name__ == "__main__":
    main()
