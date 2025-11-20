# ======================================================================================
# Standalone Inference Script
#
# Description:
# This script loads a fully merged, fine-tuned model and runs inference on it.
# It is designed for quick testing and validation of the model's performance
# after the merging process is complete.
#
# Workflow:
# 1. Load Model and Tokenizer: The script loads the specified model and tokenizer
#    from the directory where the merged model was saved.
# 2. Format Prompt: It takes a user-provided question and formats it into the
#    specific prompt structure that the model was trained on.
# 3. Generate Response: The model generates a response to the prompt.
# 4. Print Output: The script decodes and prints the model's response,
#    attempting to format it as pretty-printed JSON if possible.
# ======================================================================================

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

def run_inference(args):
    """
    Loads a model and runs inference on a given prompt.
    """
    print(f"Loading model from: {args.model_path}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    except OSError:
        print(f"Error: Model or tokenizer not found at path: {args.model_path}")
        print("Please ensure you have run the 'make merge-and-test' command to create the merged model.")
        return

    # Format the prompt exactly as it was during training
    prompt = f"<|user|>\n{args.question}<|end|>\n<|assistant|>\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print("\n--- Running Test Inference ---")
    print(f"Prompt:\n{prompt}")
    
    # Generate the response, disabling the cache to avoid the 'seen_tokens' bug
    outputs = model.generate(
        **inputs, 
        max_new_tokens=args.max_new_tokens, 
        pad_token_id=tokenizer.eos_token_id,
        use_cache=False
    )
    
    # Decode the output, keeping special tokens to locate the assistant's response
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # More robustly extract the assistant's response
    assistant_marker = "<|assistant|>"
    if assistant_marker in response_text:
        # Take the part after the marker
        assistant_response = response_text.split(assistant_marker, 1)[1]
        
        # Clean up any end-of-sequence tokens
        end_token_marker = "<|end|>"
        if end_token_marker in assistant_response:
            assistant_response = assistant_response.split(end_token_marker, 1)[0]

        assistant_response = assistant_response.strip()
    else:
        # Fallback if the marker isn't found
        prompt_part = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
        assistant_response = response_text[len(prompt_part):].strip()

    print("\nModel Response:")
    try:
        # Try to pretty-print if it's valid JSON
        parsed_json = json.loads(assistant_response)
        print(json.dumps(parsed_json, indent=2))
    except (json.JSONDecodeError, TypeError):
        # Otherwise, print as raw text
        print(assistant_response)
        
    print("\n--- Test Inference Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a merged fine-tuned model.")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="models/phi3_merged_model", 
        help="Path to the directory containing the merged model and tokenizer."
    )
    parser.add_argument(
        "--question", 
        type=str, 
        required=True, 
        help="The question to ask the model."
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=250, 
        help="The maximum number of new tokens to generate."
    )
    
    args = parser.parse_args()
    run_inference(args)