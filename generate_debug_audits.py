"""
Script to generate audit JSON files for the first 4 evaluation cases.
Saves them in data/debug/ for inspection and debugging.
"""

import os
import json
import yaml
from typing import Dict, Any

# Import backend modules
from backend import rag_engine
from evaluation.data_loading import load_text


def load_params() -> Dict[str, Any]:
    """Load parameters from params.yaml."""
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    """Generate audit reports for first 4 evaluation cases."""
    
    print("=" * 80)
    print("Generating Debug Audit Reports")
    print("=" * 80)
    
    # Load params
    params = load_params()
    evaluation_cases = params.get("evaluation", {}).get("ground_truth", [])
    
    # Get first 4 cases
    cases_to_process = evaluation_cases[:4]
    
    print(f"\nProcessing {len(cases_to_process)} cases:")
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
