# AI Travel Agent 🌍✈️

A multi-agent travel planning system that generates personalized itineraries, cost estimates, and venue recommendations. Built with **FastAPI**, **Ollama (Llama 3)**, and **Scikit-Learn**.

Make sure to update the "sas_token" variable in main.py in the get_data_local function which pulls from the azure container, ius private due to data privacy policy of the course

Demo to install project locally to IDE - https://youtu.be/6_4i9UUJuuo

## 📋 Prerequisites

* **Python 3.10+**
* **Ollama + 5GB memory + strong preference to use a GPU** (for local LLM inference)

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/GilCaplan/USA_trip_cost_agent
    cd USA_trip_cost_agent
    ```

2.  **Set up a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Setup Ollama:**
    * Download and install [Ollama](https://ollama.com).
    * Pull the Llama 3 model:
        ```bash
        ollama pull llama3
        ```
    * Ensure the Ollama server is running (`ollama serve`).

## Model Setup

Before running the agent, which is done automatically when running ```python3 main.py``` after installed everything, which will pull and initialize the machine learning models and data.

1.  **Models -  In Azure Container:**
    * `recommendation_engine.pkl`
    * `food_cost_predictor_v1.pkl`
    * `lodging_cost_predictor_v1.pkl`
    * `entertainment_cost_model.pkl`
    * `transport_cost_model.pkl`
    * `city_profiles.csv`

## 🚀 Usage

1.  **Start the Server:**
    ```bash
    python3 main.py
    ```
    The API will launch at `http://localhost:8000`.

2.  **Use the Frontend:**
    Open your browser to `http://localhost:8000` to interact with the travel planner.


## 📂 Project Structure (just files required for User interface)

* **`main.py`**: FastAPI app & Agent Orchestrator (entry point).
* **`travel_agent.py`**: DataHandler logic connecting ML models.
* **`sub_agents/`**: specialized logic for Food, Lodging, & Recommendations.
* **`static/`**: Frontend HTML/CSS assets.

* Notebooks that we ran in databricks to train models can be found in the notebooks_models directory
* Visualizations and code relevant in the visualization directory
