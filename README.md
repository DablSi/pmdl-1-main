# PMLDL-1

## Assignment Overview

This repository contains the solution for an assignment on the course of Practical Machine Learning and Deep Learning (PMLDL). The task involves deploying a machine learning model and creating a simple automated MLOps pipeline. The main task is to deploy a model in an API and create a web application that interacts with the API. The model API accepts requests from the web application and sends back responses. 

## Model

For this solution, I use the `distil-bert` model from HuggingFace for Named Entity Recognition (NER). The backend uses model caching, optimized Docker configuration, and health checks. The frontend inputs text from the user and annotates different entities in it. The possible entity types are:

- **MISC**: Miscellaneous entity
- **PER**: Person’s name
- **ORG**: Organization
- **B-LOC**: Location

## Repository Structure

```
PMLDL-1/
├── services/
│   ├── backend/
│   │   ├── src/
│   │   │   └── main.py
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   └── oetry.lock
│   ├── frontend/
│   │   ├── src/
│   │   │   └── main.py
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   └── poetry.lock
├── docker-compose.yml
├── .dockerignore
├── .gitignore
└── README.md
```

## Prerequisites

- Docker
- Docker Compose
- Poetry (for dependency management)

## Setup

1. **Clone the repository**:

    ```sh
    git clone https://github.com/your-username/PMLDL-1.git
    cd PMLDL-1
    ```

2. **Build and run the Docker containers**:

    ```sh
    docker-compose up --build
    ```

3. **Access the web application**:

    Open your web browser and navigate to `http://localhost:8501` to access the Streamlit frontend.

## Backend

The backend is built using FastAPI and runs the `distil-bert` model for NER. It includes:

- **Model Caching**: To improve performance by caching the model.
- **Optimized Docker Configuration**: To ensure efficient resource usage.
- **Health Checks**: To monitor the status of the API.

## Frontend

The frontend is built using Streamlit and includes:

- **Input Fields**: For users to input text.
- **Prediction Area**: To display the annotated entities.

For color annotation, I used the [st-annotated-text](https://github.com/tvst/st-annotated-text) library.

## Technologies Used

- **Docker**: For containerization.
- **FastAPI**: For building the backend API.
- **Streamlit**: For building the frontend web application.
- **HuggingFace Transformers**: For the `distil-bert` model.
- **st-annotated-text**: For color annotation in the frontend.
- **Poetry**: For dependency management.

## Acknowledgments

- Thanks to the creators of the `st-annotated-text` library for providing a simple way to annotate text in Streamlit.
- Thanks to the HuggingFace team for the `distil-bert` model and the Transformers library.
