# Sweet2Plus test/analysis environment
#
# Build:
#   docker build -t sweet2plus .
#
# Run tests inside the container:
#   docker run --rm sweet2plus
#
# Drop into a shell for interactive work:
#   docker run --rm -it sweet2plus bash
FROM python:3.10-slim

# System libraries required by opencv-python-headless, matplotlib, and suite2p.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first so Docker can cache this layer
# independently of source code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Install the package itself in editable mode.
COPY . .
RUN pip install --no-cache-dir -e .

# Run the (currently import-only) smoke test suite by default.
CMD ["pytest", "-v"]
