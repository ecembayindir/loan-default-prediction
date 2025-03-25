FROM python:3.9-slim

# Create a non-root user
RUN addgroup --system appuser && adduser --system --ingroup appuser appuser

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project, including the Models directory
COPY . .

# Verify model file exists in the image
RUN ls -l Models/

# Ensure correct permissions
RUN chown -R appuser:appuser /app
USER appuser

ENV PORT=5001

EXPOSE 5001

CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "2", "--threads", "2", "--log-level", "debug", "app:app"]