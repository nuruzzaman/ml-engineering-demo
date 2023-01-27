FROM python:3.9.16

ENV PYTHONUNBUFFERED 1

# Copy the application code
COPY . /ml-demo

# Set the working directory
WORKDIR /ml-demo

# Install dependencies
COPY requirements.txt .

RUN python -m pip install --upgrade pip -U
RUN pip install -r requirements.txt

# Expose the port for the Flask server
EXPOSE 5000

# Run the Flask server
ENTRYPOINT ["python", "app.py", "--host=0.0.0.0"]
