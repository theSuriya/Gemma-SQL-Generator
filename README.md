# Gemma 2B Fine-Tuned SQL Generator

## Introduction
The Gemma 2B SQL Generator is a specialized version of the Gemma 2B model, fine-tuned to generate SQL queries based on a given SQL context. This model has been tailored to assist developers and analysts in generating accurate SQL queries automatically, enhancing productivity and reducing the scope for errors.

## Model Details
- **Model Type:** Gemma 2B
- **Fine-Tuning Details:** The model was fine-tuned specifically for generating SQL queries.
- **Training Loss:** Achieved a training loss of 0.3, indicating a high level of accuracy in SQL query generation.
## Model Link
The model Available on HuggingFace[Model Link](https://huggingface.co/suriya7/Gemma2B-Finetuned-Sql-Generator)
## Installation
To set up the necessary environment for using the SQL Generator, run the following commands:
```bash
pip install torch torch
pip install transformers
```
To set up the necessary environment for Fine Tuning the SQL Generator, run the following commands:
```bash
!pip install -q -U datasets==2.16.1
!pip install -q -U bitsandbytes==0.42.0
!pip install -q -U peft==0.8.2
!pip install -q -U trl==0.7.10
!pip install -q -U accelerate==0.27.1
!pip install -q -U transformers==4.38.0
```
## Inference
```python
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("suriya7/Gemma2B-Finetuned-Sql-Generator")
model = AutoModelForCausalLM.from_pretrained("suriya7/Gemma2B-Finetuned-Sql-Generator")

prompt_input = input('enter the prompt')
context_input  = input('enter the context')

prompt_template = """
<start_of_turn>user
You are an intelligent AI specialized in generating SQL queries.
Your task is to assist users in formulating SQL queries to retrieve specific information from a database.
Please provide the SQL query corresponding to the given prompt and context:

Prompt:
{prompt}

Context:
{content}<end_of_turn>
<start_of_turn>model
"""

prompt = prompt_template.format(prompt=prompt_input,context=context_input)
encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
inputs = encodeds.to(device)


# Increase max_new_tokens if needed
generated_ids = model.generate(inputs, max_new_tokens=1000, do_sample=True, temperature = 0.7,pad_token_id=tokenizer.eos_token_id)
ans = ''
for i in tokenizer.decode(generated_ids[0], skip_special_tokens=True).split('<end_of_turn>')[:2]:
    ans += i

# Extract only the model's answer
model_answer = ans.split("model")[1].strip()
print(model_answer)
```
