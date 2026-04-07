import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
import torch.nn.functional as F

app = FastAPI(title="Gemma 4-31B API")

# Configuration
MODEL_ID = os.getenv("MODEL_ID", "/app")  # Use local files in /app
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# QUANTIZATION can be "none", "4bit", "8bit"
QUANTIZATION = os.getenv("QUANTIZATION", "none").lower()

print(f"Loading Gemma 4-31B-it from {MODEL_ID}...")

# Quantization config
bnb_config = None
if QUANTIZATION == "4bit":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )
elif QUANTIZATION == "8bit":
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

try:
    # Load processor and model
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        trust_remote_code=True
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    # In some environments, it might be better to download from HF if local files are missing
    # But usually, it's better to fail and let user know.

class Message(BaseModel):
    role: str
    content: Union[str, List[dict]]

class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 64
    thinking: Optional[bool] = False

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        # Prepare messages for template
        chat_messages = []
        for msg in request.messages:
            chat_messages.append({"role": msg.role, "content": msg.content})

        # Apply chat template
        prompt = processor.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=request.thinking
        )

        # Tokenize
        inputs = processor(text=prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                do_sample=True if request.temperature > 0 else False
            )

        # Decode
        response_text = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
        parsed_response = processor.parse_response(response_text)

        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": parsed_response.get("content", response_text)
                },
                "thought": parsed_response.get("thought", "")
            }],
            "model": MODEL_ID
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
