import json
import argparse
import sys
import os
import io
import contextlib
from typing import List, Dict, Any

# Ensure we can import from the root directory
sys.path.append(os.getcwd())

# Try importing Ragas
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from datasets import Dataset
except ImportError:
    print("❌ Error: 'ragas' or 'datasets' package not found.")
    print("Please install them using: pip install ragas datasets pandas")
    print("Note: You also need an OpenAI API key set in your environment.")
    sys.exit(1)

# Try importing system functions
try:
    # Import get_context or query_memory. 
    # agent.py imports get_context from rag_v2.query. We will use query_memory to get structured contexts.
    from rag_v2.query import query_memory
    from langraph.agent import run_agent
except ImportError as e:
    print(f"❌ Error importing system functions: {e}")
    print("Make sure you are running this from the project root directory context.")
    sys.exit(1)


def get_rag_response(question: str) -> Dict[str, Any]:
    """
    Get the answer and context from the existing RAG system.
    We suppress the standard output of the underlying system to keep the console clean,
    unless an error occurs.
    """
    
    # We'll use a string buffer to capture output in case we need to debug/show it on error
    f = io.StringIO()
    
    try:
        # Get Answer using the agent
        # The agent uses get_context internally with k=3
        with contextlib.redirect_stdout(f):
            answer = run_agent(question)
            
            # Get Context explicitly to pass to Ragas
            # We use the same k=3 as the agent
            results = query_memory(question, k=3)
            contexts = [r.get("text", "") for r in results]
            
        return {
            "answer": answer,
            "contexts": contexts
        }
    except Exception as e:
        # If something failed, print the captured logs for debugging
        print(f.getvalue())
        print(f"❌ Error processing question '{question}': {e}")
        return {
            "answer": "Error generating answer.",
            "contexts": []
        }

def run_evaluation(test_file: str, output_file: str):
    """
    Run the RAGAS evaluation pipeline.
    """
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        sys.exit(1)
        
    print(f"📄 Loading test cases from {test_file}...")
    try:
        with open(test_file, 'r') as f:
            test_cases = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load JSON file: {e}")
        sys.exit(1)

    if not isinstance(test_cases, list):
        print("❌ Test file must contain a list of objects.")
        sys.exit(1)

    # Prepare data for RAGAS
    data_samples = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    print(f"🔄 Processing {len(test_cases)} test cases...")
    print("-" * 50)

    for i, case in enumerate(test_cases):
        q = case.get("question")
        gt = case.get("ground_truth") or case.get("answer") # Fallback to 'answer' if grounded truth labeled differently
        
        if not q:
            print(f"⚠️  Skipping case {i+1}: Missing question")
            continue
            
        print(f"[{i+1}/{len(test_cases)}] ❓ {q}")
        
        # Call system
        response = get_rag_response(q)
        
        data_samples["question"].append(q)
        data_samples["answer"].append(response["answer"])
        data_samples["contexts"].append(response["contexts"])
        data_samples["ground_truth"].append(gt if gt else "")

    # Convert to Dataset
    dataset = Dataset.from_dict(data_samples)
    
    print("\n📈 Running RAGAS metrics...")
    
    # Configure Ragas with explicit LLM/Embeddings to avoid default errors
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        
        # Initialize models
        # Use gpt-4o-mini for evaluation as it is cost effective and capable
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
        evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
        
        # Run evaluation with explicit config
        results = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            raise_exceptions=False 
        )
    except Exception as e:
        print(f"⚠️  Could not configure custom LLM/Embeddings, trying defaults... Error: {e}")
        # Fallback to defaults
        try:
            results = evaluate(
                dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ],
                raise_exceptions=False
            )
        except Exception as e2:
             print(f"❌ RAGAS evaluation failed: {e2}")
             sys.exit(1)
    
    # Print clean summary
    print("\n" + "="*50)
    print("📊 RAGAS EVALUATION SUMMARY")
    print("="*50)
    
    # Handle results display
    if hasattr(results, 'items'): # Result object
        for metric, score in results.items():
            print(f"{metric:<20}: {score:.4f}")
    else:
        print(results)
    
    print("="*50)
    
    # Save raw output
    # Save raw output
    try:
        final_output = {}
        
        # Handle Summary - Try conversion to dict
        try:
             # Try to convert to dict (works for most Mappings)
             final_output["summary"] = dict(results)
        except Exception:
             # Fallback: maybe it's a dataframe or custom object, try pandas mean or string
             try:
                 if hasattr(results, 'to_pandas'):
                     final_output["summary"] = results.to_pandas().mean(numeric_only=True).to_dict()
                 else:
                     final_output["summary"] = str(results)
             except:
                 final_output["summary"] = str(results)
             
        # Handle Details (per sample)
        if hasattr(results, 'to_pandas'):
            res_df = results.to_pandas()
            # Convert to list of dicts, ensuring primitives
            details = res_df.to_dict(orient="records")
            final_output["details"] = details
        else:
            final_output["details"] = "Detailed per-sample metrics not available (results object mismatch)"
        
        with open(output_file, 'w') as f:
            json.dump(final_output, f, indent=2, default=str)
            
        print(f"\n✅ Detailed results saved to: {output_file}")
        
    except Exception as e:
        print(f"⚠️  Failed to save results: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on the RAG system")
    parser.add_argument("test_file", help="Path to JSON file containing test cases (must have 'question' and 'ground_truth')")
    parser.add_argument("--output", "-o", default="ragas_results.json", help="Path to save the output JSON")
    
    args = parser.parse_args()
    
    print("🚀 Starting RAGAS Evaluation Module")
    run_evaluation(args.test_file, args.output)
