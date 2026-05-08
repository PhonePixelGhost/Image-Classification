FROM python:3.11-slim

# Create a user to avoid running as root (Required for Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

WORKDIR /app

# Copy requirements and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the application code and models
COPY --chown=user app/ ./app/
COPY --chown=user models/ ./models/

# Use port 7860 (Standard for Hugging Face Spaces)
EXPOSE 7860

# Run with multiple workers for higher throughput
# Note: On Linux/Docker, uvicorn workers work perfectly
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "4", "--log-level", "warning"]
