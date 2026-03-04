"""
Script to generate audit JSON files for evaluation cases.
Saves them in data/debug/ for inspection and debugging.

Usage:
    python generate_debug_audits.py                    # Process first 5 cases (default)
    python generate_debug_audits.py --case 0           # Process case at index 0
    python generate_debug_audits.py --case Evaluation1 # Process case by name
    python generate_debug_audits.py --case all         # Process all cases
"""

import os
import json
import yaml
import argparse
from typing import Dict, Any, List

# Import backend modules
from backend import rag_engine
from evaluation.data_loading import load_text


def load_params() -> Dict[str, Any]:
    """Load parameters from params.yaml."""
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate audit JSON files for evaluation cases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_debug_audits.py                    # Process first 5 cases (default)
  python generate_debug_audits.py --case 0           # Process case at index 0
  python generate_debug_audits.py --case Evaluation1 # Process case by name
  python generate_debug_audits.py --case all         # Process all cases
        """
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Case to process: index (0-based), name, or 'all' for all cases. Default: first 5 cases"
    )
    return parser.parse_args()


def select_cases(evaluation_cases: List[Dict[str, Any]], case_arg: str = None) -> List[Dict[str, Any]]:
    """
    Select evaluation cases based on command line argument.
    
    Args:
        evaluation_cases: Full list of evaluation cases from params.yaml
        case_arg: Command line argument: index, name, 'all', or None for first 5
        
    Returns:
        List of selected cases to process
    """
    if case_arg is None:
        # Default: first 5 cases (Evaluation1-5)
        return evaluation_cases[:5]
    
    if case_arg.lower() == "all":
        # Process all cases
        return evaluation_cases
    
    # Try to parse as index
    try:
        index = int(case_arg)
        if 0 <= index < len(evaluation_cases):
            return [evaluation_cases[index]]
        else:
            print(f"❌ Error: Index {index} out of range (0-{len(evaluation_cases)-1})")
            return []
    except ValueError:
        pass
    
    # Try to match by name
    for case in evaluation_cases:
        if case.get("name") == case_arg:
            return [case]
    
    # Case not found
    print(f"❌ Error: Case '{case_arg}' not found")
    print(f"\nAvailable cases:")
    for idx, case in enumerate(evaluation_cases):
        print(f"  [{idx}] {case.get('name')}")
    return []


def main():
    """Generate audit reports for selected evaluation cases."""
    
    # Parse arguments
    args = parse_args()
    
    print("=" * 80)
    print("Generating Debug Audit Reports")
    print("=" * 80)
    
    # Load params
    params = load_params()
    evaluation_cases = params.get("evaluation", {}).get("ground_truth", [])
    
    print(f"\nTotal available cases: {len(evaluation_cases)}")
    
    # Select cases to process
    cases_to_process = select_cases(evaluation_cases, args.case)
    
    if not cases_to_process:
        print("\n❌ No cases selected. Exiting.")
        return
    
    print(f"\nProcessing {len(cases_to_process)} case(s):")
    for case in cases_to_process:
        print(f"  - {case.get('name')}")
    
    # Initialize RAG engine
    print("\n🔧 Initializing RAG engine...")
    rag_engine.init_rag()
    print("✓ RAG engine initialized")
    
    # Create debug directory if it doesn't exist
    debug_dir = "data/debug"
    os.makedirs(debug_dir, exist_ok=True)
    print(f"✓ Debug directory: {debug_dir}")
    
    # Process each case
    for i, case in enumerate(cases_to_process, 1):
        case_name = case.get("name")
        doc_path = case.get("document_path")
        
        print(f"\n{'='*80}")
        print(f"[{i}/{len(cases_to_process)}] Processing: {case_name}")
        print(f"{'='*80}")
        
        if not os.path.exists(doc_path):
            print(f"⚠️  Document not found: {doc_path}")
            continue
        
        # Load document
        print(f"📄 Loading document: {doc_path}")
        try:
            document_text = load_text(doc_path)
            print(f"✓ Document loaded ({len(document_text)} characters)")
        except Exception as e:
            print(f"❌ Failed to load document: {e}")
            continue
        
        # Run audit (full audit, no requirement limit)
        print(f"🔍 Running full audit...")
        try:
            audit_response = rag_engine.audit_document(
                document_text, 
                requirement_limit=None  # Full audit
            )
            print(f"✓ Audit completed: {len(audit_response.requirements)} requirements")
        except Exception as e:
            print(f"❌ Audit failed: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Convert to dict and save
        output_filename = f"audit_{case_name}.json"
        output_path = os.path.join(debug_dir, output_filename)
        
        print(f"💾 Saving to: {output_path}")
        try:
            audit_dict = audit_response.model_dump()
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(audit_dict, f, indent=2, ensure_ascii=False)
            
            file_size = os.path.getsize(output_path)
            print(f"✓ Saved ({file_size:,} bytes)")
            
            # Print summary
            print(f"\n📊 Summary for {case_name}:")
            for req in audit_response.requirements:
                req_id = req.Requirement_ID
                score = req.Score
                num_subs = len(req.SubRequirements) if req.SubRequirements else 0
                print(f"  • {req_id}: Score={score}, SubReqs={num_subs}")
                
        except Exception as e:
            print(f"❌ Failed to save: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print("✅ All audits completed!")
    print(f"{'='*80}")
    print(f"\nAudit files saved in: {os.path.abspath(debug_dir)}")
    print("\nGenerated files:")
    for case in cases_to_process:
        filename = f"audit_{case.get('name')}.json"
        filepath = os.path.join(debug_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"  ✓ {filename} ({size:,} bytes)")
        else:
            print(f"  ✗ {filename} (not created)")


if __name__ == "__main__":
    main()
