"""
LoRA Fine-Tuning Demo
CodeGen-350M model trained on 18k Python examples
Group 6
"""

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time
import json

st.set_page_config(page_title="LoRA Code Generator", layout="wide")

# Simple title
st.title("LoRA Code Generator Demo")
st.write("Comparing base CodeGen-350M vs fine-tuned model on Python code generation")


# Load training stats if available
@st.cache_data
def load_training_info():
    try:
        with open("./models/lora_adapters/training_info.json", "r") as f:
            return json.load(f)
    except:
        return None


training_info = load_training_info()

# Show basic stats at the top
if training_info:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Training Samples", f"{training_info.get('num_samples', 0):,}")
    col2.metric("Final Loss", f"{training_info.get('final_eval_loss', 0):.4f}")
    col3.metric("LoRA Rank", f"r={training_info.get('lora_r', 32)}")
    trainable = training_info.get('trainable_params', 0)
    total = training_info.get('total_params', 357000000)
    pct = (trainable / total) * 100 if total > 0 else 0
    col4.metric("Trainable Params", f"{pct:.2f}%")
    st.divider()


# Load models once and cache them
@st.cache_resource
def load_models():
    base_model_name = "Salesforce/codegen-350M-mono"
    adapter_path = "./models/lora_adapters"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    )

    # Load fine-tuned model with LoRA adapters
    finetuned_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    )
    finetuned_model = PeftModel.from_pretrained(finetuned_model, adapter_path)

    base_model.to(device)
    finetuned_model.to(device)
    base_model.eval()
    finetuned_model.eval()

    return tokenizer, base_model, finetuned_model, device


def generate_code(model, tokenizer, prompt, device, temperature, max_tokens, top_p, top_k):
    # Format the input with instruction template
    formatted_input = f"### Instruction:\n{prompt}\n\n### Response:\n"
    inputs = tokenizer(formatted_input, return_tensors="pt", padding=True, truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )

    # Decode and extract just the response part
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Response:" in response:
        code = response.split("### Response:")[1].strip()
        if "### Instruction:" in code:
            code = code.split("### Instruction:")[0].strip()
        return code
    return response


# Try to load models
try:
    tokenizer, base_model, finetuned_model, device = load_models()
    st.success(f"Models loaded successfully (running on {device.upper()})")
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# Generation settings in sidebar
with st.sidebar:
    st.subheader("Generation Settings")
    temperature = st.slider("Temperature", 0.1, 1.0, 0.7, 0.05)
    max_tokens = st.slider("Max Tokens", 50, 400, 200, 50)
    top_p = st.slider("Top P", 0.5, 1.0, 0.9, 0.05)
    top_k = st.slider("Top K", 10, 100, 50, 10)

# Sample prompts for easy testing
st.subheader("Code Generation")

sample_prompts = [
    "Write a Python function to calculate factorial using recursion",
    "Create a function to check if a string is palindrome",
    "Write code to merge two sorted lists",
    "Implement binary search algorithm",
    "Create a class for a stack data structure",
    "Write a function to find all prime numbers up to n"
]

prompt_choice = st.selectbox("Select a sample prompt or write your own:", ["Custom"] + sample_prompts)

if prompt_choice == "Custom":
    user_prompt = st.text_area("Enter your instruction:", height=100)
else:
    user_prompt = prompt_choice

# Generate button
if st.button("Generate Code", type="primary"):
    if user_prompt.strip():
        # Two columns for side-by-side comparison
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Base Model")
            start = time.time()
            base_output = generate_code(base_model, tokenizer, user_prompt, device,
                                        temperature, max_tokens, top_p, top_k)
            base_time = time.time() - start
            st.code(base_output, language="python")
            st.caption(f"Time: {base_time:.2f}s | Tokens: ~{len(base_output.split())}")

        with col2:
            st.subheader("Fine-tuned Model (LoRA)")
            start = time.time()
            ft_output = generate_code(finetuned_model, tokenizer, user_prompt, device,
                                      temperature, max_tokens, top_p, top_k)
            ft_time = time.time() - start
            st.code(ft_output, language="python")
            st.caption(f"Time: {ft_time:.2f}s | Tokens: ~{len(ft_output.split())}")

    else:
        st.warning("Please enter a prompt")

# Show some info at the bottom
st.divider()
st.caption("Group 6: Model Adaptation, Efficient Fine-Tuning & Deployment of LLMs")
st.caption("CodeGen-350M fine-tuned with LoRA (r=32) on 18,000 Python code examples")