from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import ollama
import json
import re
import os
import logging
from contextlib import asynccontextmanager
from travel_agent import DataHandler
import ast
import pandas as pd
import io
from azure.storage.blob import BlobServiceClient

# --- 1. ROBUST LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

data_store = {}


# Add this at the top of main.py
def extract_state(val):
    if not val: return "NY"
    s = str(val).strip().upper()
    # If it's already a code, just return it
    if len(s) == 2 and s.isalpha(): return s
    # Map for full names
    state_map = {"ALABAMA": "AL", "CALIFORNIA": "CA", "NEW YORK": "NY", "TEXAS": "TX"}  # Add more...
    return state_map.get(s, "NY")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("--- SERVER STARTUP ---")
    try:
        # 1. Download data to a specific local folder
        models_dir = os.path.abspath("./tmp/models")
        state_map, abbrev_map = get_data_local()  # Uses your Azure logic

        # 2. Instantiate Handler with the explicit path
        data_store['handler'] = DataHandler(models_path=models_dir)
        logger.info("✅ DataHandler initialized with local models")
    except Exception as e:
        logger.error(f"❌ Startup Error: {e}")
    yield

app = FastAPI(lifespan=lifespan)

# Serve the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def read_root():
    return FileResponse('static/index.html')


class TripRequest(BaseModel):
    prompt: str
    dining_ratio: int = 50  # Default to 50% if not provided by UI slider


class TripResponse(BaseModel):
    city: str
    state: str
    preferences: dict
    total_cost: float
    breakdown: dict
    food_options: dict
    lodging_details: dict
    ai_message: str


@app.post("/plan_trip", response_model=TripResponse)
async def plan_trip(request: TripRequest):
    handler = data_store['handler']

    # --- 1. ROBUST EXTRACTION STEP ---
    # Simplified Prompt: No longer asking LLM to guess ratios.
    extract_prompt = f"""
    You are a Data Extraction Specialist. Analyze this unstructured travel request: "{request.prompt}"

    TASKS:
    1. Correct any misspellings (e.g. "Pheonix" -> "Phoenix").
    2. Infer the 2-letter State Code if missing (e.g. "Chicago" -> "IL").
    3. Infer Interests: If not explicit, guess based on city vibe.
    4. Normalize Budget: Must be "Cheap", "Moderate", or "Luxury". Default "Moderate".
       - NOTE: Handle negations! "Not too expensive" -> "Cheap". "Not cheap" -> "Moderate".
    5. Infer Demographics: Estimate composition (males, females, children). 
       - CRITICAL: "4 friends" means TOTAL 4 people (e.g. 2 males, 2 females). Do NOT double count.
    6. Cuisine: Extract food types (e.g. "Italian", "Vegan"). 
       - CRITICAL: Do NOT put "cooking", "eating out", or percentages here.

    7. PRIMARY DESTINATION: If multiple cities are mentioned, choose the MAIN city.

    OUTPUT JSON ONLY (No text before or after):
    {{
      "city": "String",
      "state": "2-letter String",
      "days": Integer (Default 3),
      "demographics": {{ "males": Integer, "females": Integer, "children": Integer }},
      "cuisine": "String (or List of Strings)",
      "budget_level": "String",
      "interests": ["String", "String"]
    }}

    CRITICAL INSTRUCTIONS:
    - Output ONLY valid JSON.
    - NO conversational filler.
    - NO trailing commas.
    - Use DOUBLE QUOTES for all keys and string values.
    """

    params = {}

    try:
        response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': extract_prompt}])
        content = response['message']['content']

        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            raw_string = match.group().strip()
            # Clean trailing commas
            raw_string = re.sub(r',\s*([\]}])', r'\1', raw_string)

            try:
                params = json.loads(raw_string)
            except json.JSONDecodeError:
                try:
                    params = ast.literal_eval(raw_string)
                except Exception as e:
                    logger.error(f"Failed to parse via AST: {e}")

            logger.info(f"✅ Extracted Params: {params}")
        else:
            logger.warning("⚠️ No JSON found in LLM response")
    except Exception as e:
        logger.error(f"❌ Extraction Logic Error: {e}")

    # --- 2. VARIABLE SANITIZATION ---

    # City/State Handling (List vs String)
    raw_city = params.get('city')
    if isinstance(raw_city, list):
        raw_city = raw_city[0] if len(raw_city) > 0 else 'New York'
    city = str(raw_city or 'New York').strip().title()

    raw_state = params.get('state')
    if isinstance(raw_state, list):
        raw_state = raw_state[0] if len(raw_state) > 0 else 'NY'
    state = str(raw_state or 'NY').strip().upper()[:2]

    try:
        days = int(params.get('days') or 3)
        if days < 1: days = 3
    except:
        days = 3

    # Demographics
    raw_demo = params.get('demographics') or {}
    demographics = {
        "males": int(raw_demo.get('males', 1)),
        "females": int(raw_demo.get('females', 0)),
        "children": int(raw_demo.get('children', 0))
    }
    if sum(demographics.values()) < 1:
        demographics["males"] = 1
    ppl = sum(demographics.values())

    # --- DETERMINISTIC RATIO LOGIC ---
    # Directly use the slider value from the request. No LLM involvement.
    slider_ratio = max(0, min(100, request.dining_ratio)) / 100.0
    logger.info(f"⚖️ Using Slider Ratio: {slider_ratio}")

    # --- CUISINE PARSING ---
    def format_cuisine(val):
        if not val:
            return "American"
        items = val if isinstance(val, list) else [val]

        # Filter out "ratio" garbage that might still sneak in (e.g. "35%", "cooking")
        cleaned = []
        for i in items:
            s = str(i).strip()
            # If it looks like a number/percentage, skip it
            if re.search(r'\d+%', s) or s.lower() in ['cooking', 'eating out']:
                continue
            if s:
                cleaned.append(s)

        return ", ".join(cleaned) if cleaned else "American"

    cuisine = format_cuisine(params.get('cuisine'))

    # --- BUDGET MAPPING WITH NEGATION LOGIC ---
    def map_budget(input_text, full_prompt):
        # Normalize inputs
        raw_budget = str(input_text or "moderate").lower()
        prompt_text = full_prompt.lower()

        # 1. Check for specific negation phrases in the full prompt
        cheap_triggers = [
            "not too expensive", "not expensive", "not very expensive",
            "not that expensive", "not pricey", "inexpensive", "fair price",
            "good price", "reasonable", "affordable", "not to expensive",
            "not to pricy", "on a budget"
        ]

        moderate_triggers = [
            "not cheap", "not too cheap", "not very cheap",
            "not the cheapest", "mid range", "middle ground",
            "average price", "not low cost", "Not Cheap"
        ]

        # Check for "Cheap" indicators first (negated expensive)
        if any(phrase in str(prompt_text).lower() for phrase in cheap_triggers):
            return "Cheap"

        # Check for "Moderate/Luxury" indicators (negated cheap)
        if any(phrase in str(prompt_text).lower() for phrase in moderate_triggers):
            return "Moderate"

            # 2. Standard Keyword Mapping
        mapping = {
            "Cheap": ["cheap", "budget", "low", "affordable", "frugal", "economical", "typical"],
            "Moderate": ["moderate", "medium", "middle", "average", "standard", "comfortable"],
            "Luxury": ["luxury", "expensive", "high", "premium", "top-tier", "baller", "extravagant"]
        }

        # Check if any keyword exists in the extracted budget field
        for category, keywords in mapping.items():
            if any(word in raw_budget for word in keywords):
                return category

        return "Moderate"

    budget = map_budget(params.get('budget_level'), request.prompt)

    interests = params.get('interests') or []
    if not isinstance(interests, list) or not interests:
        interests = ["Tourism", "Landmarks", "Food"]

    # --- 3. COST ESTIMATION ---
    logger.info(f"🔮 Predicting costs for {city}, {state} ({budget})...")

    costs = handler.get_trip_estimate(
        city, state, days,
        travelers=ppl,
        demographics=demographics,
        eating_out_ratio=slider_ratio,
        cuisine=cuisine,
        budget_level=budget
    )

    # --- 4. SMART RECOMMENDATION (Closed Loop) ---
    try:
        predicted_ent_budget = float(costs['breakdown'].get('Entertainment', 100.0))
    except:
        predicted_ent_budget = 100.0
        logger.info("error on recommendation system")

    rec_prefs = {
        'interests': interests,
        'budget': budget,
        'max_spend': predicted_ent_budget,
        'travelers': ppl
    }

    logger.info(f"🔍 Running Recommender with budget ${predicted_ent_budget}...")
    raw_recs = handler.get_recommendations(city, state, rec_prefs)

    if raw_recs:
        rec_text = "\n".join([f"- {r['name']} ({r['category'].title()}, {r['price'].title()} tier)" for r in raw_recs])
    else:
        rec_text = f"Suggest general popular activities in {city} suitable for a {budget} budget."

    # --- 5. NARRATIVE GENERATION ---
    narrative_prompt = f"""
    You are a Strategic Travel Consultant. Summarize this trip professionally.

    ### TRIP DATA
    - **Destination:** {city}, {state} for {days} days
    - **Group:** {ppl} people ({demographics['males']}M, {demographics['females']}F, {demographics['children']} Kids)
    - **Food Plan:** Eating out {int(slider_ratio * 100)}% of the time.
    - **Total Cost:** ${costs['total']}

    ### TOP-RANKED VENUES
    {rec_text}

    ### INSTRUCTIONS
    1. **Overview:** A 1-sentence hook about {city}.
    2. **Strategic Cost Analysis:** Why does this trip cost ${costs['total']}? Mention how their group size ({ppl}) and dining ratio ({int(slider_ratio * 100)}% restaurants) impacts the price.
    3. **Activity Justification:** Briefly explain the venue choices.
    4. **Final Breakdown:** Provide a clear list of costs.

    **FORMATTING:** Use professional Markdown, **bold** key figures, and headers (###).
    """

    try:
        final_res = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': narrative_prompt}])
        ai_message = final_res['message']['content']
    except Exception as e:
        logger.error(f"Narrative Generation Error: {e}")
        ai_message = "Error generating narrative. Please check the breakdown above."

    return {
        "city": city,
        "state": state,
        "preferences": {
            "cuisine": cuisine,
            "style": budget,
            "interests": interests,
            "group_size": ppl,
            "dining_ratio": slider_ratio
        },
        "total_cost": costs['total'],
        "breakdown": costs['breakdown'],
        "food_options": costs['food_options'],
        "lodging_details": costs.get('lodging_details', {}),
        "ai_message": ai_message
    }


def get_data_local():
    # Setup configuration
    storage_account_url = "https://lab94290.blob.core.windows.net"
    # Ensure sas_token includes the '?' at the start
    sas_token = "ADD"

    container_name = "submissions"
    group = "Gil_Murad_Guy"

    # Authenticate specifically for SAS tokens
    blob_service_client = BlobServiceClient(account_url=storage_account_url, credential=sas_token)
    container_client = blob_service_client.get_container_client(container_name)

    # Use a single local folder for all agent resources
    models_dir = os.path.abspath("./tmp/models")
    os.makedirs(models_dir, exist_ok=True)

    print("--- Syncing Models & Transportation Data ---")
    all_blobs = container_client.list_blobs(name_starts_with=f"{group}/")
    for blob in all_blobs:
        filename = os.path.basename(blob.name)

        # Existing renaming logic...
        if filename == "city_profiles.parquet":
            # Download as parquet, then convert to CSV for the agent
            temp_parquet = os.path.join(models_dir, "city_profiles.parquet")
            with open(temp_parquet, "wb") as f:
                f.write(container_client.download_blob(blob).readall())

            # Convert to CSV so the EntertainmentAgent can find it
            pd.read_parquet(temp_parquet).to_csv(os.path.join(models_dir, "city_profiles.csv"), index=False)
            print("✅ Converted city_profiles.parquet to .csv for Entertainment Agent")
            continue
    for blob in all_blobs:
        if blob.name.endswith(".crc"): continue

        filename = os.path.basename(blob.name)

        # Mapping cloud names to TransportAgent's hardcoded expectations
        if filename == "cities_us.csv":
            target_name = "transportation_data.csv"
        elif filename == "transportation_state.json":
            target_name = "transportation_state.json"
        else:
            target_name = filename

        local_path = os.path.join(models_dir, target_name)
        with open(local_path, "wb") as f:
            f.write(container_client.download_blob(blob).readall())
        print(f"✅ Synced: {target_name}")



    # Return state mappings for the FastAPI data_store
    # We use transportation_state.json if it exists, otherwise fallback to states.csv
    return {}, {}  # Or implement your mapping logic here

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)