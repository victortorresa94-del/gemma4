# Use an official NVIDIA CUDA image as the base
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Update and install dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    libgl1-mesa-glx \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Copy the requirements file and install Python packages
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Install flash-attention (optional but recommended for Gemma)
RUN python3 -m pip install --no-cache-dir flash-attn --no-build-isolation

# Copy the rest of the repository contents
COPY . .

# Environment variable for Hugging Face (optional - user can set at runtime)
# ENV HF_TOKEN=your_token_here

# Expose port (if using FastAPI/serving)
EXPOSE 8000

# Entrypoint for RunPod (you can customize this to run your serving script)
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
