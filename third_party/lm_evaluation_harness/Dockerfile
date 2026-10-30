FROM python:3.11-slim

# Install first to cache the dependencies
RUN pip install --no-cache-dir 'lm_eval[math,openai]==0.4.3' --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple

WORKDIR /lm-evaluation-harness

# deprecated name
ENV TRANSFORMERS_CACHE=/lm-evaluation-harness/hf
# new env name
ENV HF_HOME=/lm-evaluation-harness/hf

# Set to EMPTY for local evaluation
ENV OPENAI_API_KEY=EMPTY

RUN mkdir -p /lm-evaluation-harness/hf

## Non cached section starts here

COPY . /lm-evaluation-harness

RUN pip install --no-cache-dir '.[math,openai]' --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple

RUN python nvbench/fetch_datasets.py
