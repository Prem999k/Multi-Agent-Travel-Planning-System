# ✈️ TravelMind AI — A Multi-Agent Travel Planner with LangGraph, MCP & Human-in-the-Loop

TravelMind AI is a real-world, multi-agent travel planning system that converts a natural-language trip request into a practical, budget-aware travel blueprint.

🔗 Live Demo

https://multi-agent-travel-planning-system-x2u1.onrender.com

Instead of asking one LLM to do everything, the system uses **LangGraph orchestration**, specialist agents, **MCP-based travel tools**, **PostgreSQL checkpointing**, **input guardrails**, **dynamic supervisor routing**, and a **Human-in-the-Loop (HITL)** approval step to build and refine a final travel plan.

The application provides a web interface backed by FastAPI and produces a structured travel response covering trip summary, financial planning, day-by-day itinerary, money-saving tips, reminders, packing guidance, and final recommendations.

---

## 🌍 What Problem Does It Solve?

Trip planning normally requires switching between flight websites, hotel-search platforms, weather services, maps, budgeting tools, and notes.

TravelMind AI brings these tasks into one coordinated workflow:

```text
Natural-Language Trip Request
            ↓
      Input Guardrail
            ↓
       Supervisor
            ↓
   Dynamic Agent Selection
            ↓
 ┌──────────┼──────────┬───────────┐
 ↓          ↓          ↓           ↓
Flight     Hotel      Weather     Budget
Agent      Agent      Agent       Agent
 └──────────┴──────────┴───────────┘
            ↓
     Itinerary Agent
            ↓
   Draft Travel Blueprint
            ↓
 Human-in-the-Loop Review
       ↙           ↘
  Revision         Approval
      ↓               ↓
      └───────┬───────┘
              ↓
       Final Response Agent
              ↓
       Final Travel Blueprint
```

---

## ✨ Key Features

- ✈️ **Flight planning** using AviationStack through MCP
- 🏨 **Hotel and accommodation research** using Tavily MCP search
- 🌦️ **Current weather and forecast** using a custom OpenWeather MCP server
- 🧭 **Supervisor Agent** for dynamic workflow routing
- 🛡️ **Input Guardrail** to validate travel-related requests
- 💰 **Budget Agent** for total-group budget analysis and feasibility
- 🗓️ **Itinerary Agent** for the main travel blueprint
- 👤 **Human-in-the-Loop approval** before finalization
- 🔁 **Revision through human feedback**
- 💾 **PostgreSQL persistence** through LangGraph `PostgresSaver`
- ⚡ **Groq-powered LLM inference** using `openai/gpt-oss-20b`
- 🌐 **FastAPI backend** with HTML/CSS/JavaScript frontend
- 📄 **Copy and PDF export** from the web interface
- 🐳 **Docker-ready deployment**
- 🔐 Environment-based API key configuration

---

## 🔗 Live Demo

```text
https://multi-agent-travel-planning-system-x2u1.onrender.com
```

# 🧠 End-to-End Workflow

This section describes what happens from the moment a user submits a request until the final travel plan is displayed.

## 1. User submits a natural-language request

The user enters a request such as:

```text
Plan a complete 5 days Tirupati trip from Hyderabad for 4 people,
including travel, hotels and sightseeing under ₹25,000 total.
```

The frontend sends the request to:

```text
POST /api/travel
```

A `thread_id` can also be supplied so the same LangGraph conversation/checkpoint can be resumed later.

---

## 2. FastAPI receives the request

`app.py` validates the request and calls:

```python
run_travel_agent(user_input=user_message, thread_id=request_data.thread_id)
```

The backend then starts or resumes the LangGraph workflow.

The application also exposes:

```text
GET  /health
POST /api/travel
POST /api/travel/approve
```

---

## 3. Input Guardrail validates the request

The first logical stage is the **Input Guardrail**, implemented inside the Supervisor node.

It checks whether the request is related to legitimate travel planning or travel information, such as:

- destinations
- flights
- hotels
- weather
- budgets
- transportation
- sightseeing
- food
- packing
- visas
- itineraries

Clearly unrelated or harmful/illegal requests are blocked.

For a valid request, the guardrail allows the workflow to continue.

If the guardrail blocks a request, the workflow routes to the blocked-response node and terminates without executing the travel specialist agents.

---

## 4. Supervisor Agent interprets the trip request

After guardrail validation, the **Supervisor Agent** determines which specialist agents are required.

It extracts structured trip constraints such as:

```text
Destination
Origin
Duration
Budget
Travel style
Special preferences
```

The Supervisor returns structured JSON and selects from:

```text
flight_agent
hotel_agent
weather_agent
budget_agent
itinerary_agent
```

The `itinerary_agent` is always required because it produces the integrated travel blueprint.

### Why the Supervisor matters

The Supervisor prevents every request from blindly running the same fixed workflow. Agent execution can be selected dynamically according to the request.

For example:

```text
Flight + Hotel request
        ↓
Flight Agent
Hotel Agent
Itinerary Agent
```

while a broader trip request may use:

```text
Flight
Hotel
Weather
Budget
Itinerary
```

If Supervisor parsing fails, the backend safely falls back to the complete specialist workflow instead of abandoning the request.

---

# 🔧 Specialist Agents

## 5. Flight Agent

The Flight Agent gathers aviation information through the AviationStack MCP integration.

It obtains information such as:

- airports
- airlines
- likely departure and arrival airports
- relevant airlines
- typical flight duration
- estimated fare guidance when supported
- peak-season considerations
- practical booking advice

The agent is instructed not to claim that a flight or ticket is confirmed unless the supplied data explicitly confirms it.

### MCP path

```text
Flight Agent
    ↓
aviation_mcp_call(...)
    ↓
AviationStack MCP
    ↓
Airport / airline data
```

---

## 6. Hotel Agent

The Hotel Agent uses the Tavily MCP search integration to research:

- hotel options
- recommended areas to stay
- practical accommodation guidance

The result is passed into the downstream planning stages.

If live Tavily hotel research fails, the backend uses a fallback message and instructs the downstream planner to provide general accommodation guidance rather than pretending that live data was available.

### MCP path

```text
Hotel Agent
    ↓
tavily_mcp_search(...)
    ↓
Tavily MCP
    ↓
Accommodation research
```

---

## 7. Weather Agent

The Weather Agent extracts the destination and calls the custom weather MCP server.

It obtains:

```text
Current weather
Forecast information
```

The custom server is implemented in:

```text
custom_weather_mcp_server.py
```

It uses OpenWeather to return structured weather information.

### MCP path

```text
Weather Agent
      ↓
extract_destination()
      ↓
weather_mcp_search()
      ↓
Custom Weather MCP Server
      ↓
OpenWeather API
      ↓
Current Weather + Forecast
```

If live weather is unavailable, the system falls back to seasonal guidance and tells the final planner to verify the forecast before departure.

---

## 8. Budget Agent

The Budget Agent evaluates whether the requested trip is realistic for the user's **total group budget**.

This distinction is important:

```text
₹25,000 for 4 people
        ≠
₹25,000 per person
```

The agent considers:

- traveler count
- flights/intercity transport
- accommodation
- local transport
- food and miscellaneous expenses
- activities
- budget risk areas
- money-saving opportunities

### Accommodation rule

The current planning logic uses:

```text
1 room  → 4 or fewer travelers
2 rooms → more than 4 travelers
```

An explicit user room request overrides the default rule.

---

# 🗓️ Itinerary Generation

## 9. Itinerary Agent creates the draft blueprint

The Itinerary Agent combines the information gathered by the specialist agents:

```text
User Request
     +
Trip Constraints
     +
Flight Research
     +
Hotel Research
     +
Weather Research
     +
Budget Analysis
     ↓
Itinerary Agent
     ↓
Draft Travel Blueprint
```

The planner is instructed to keep the requested duration and produce a practical chronological travel flow.

The final blueprint uses this structure:

```text
# Destination / Trip Name

## X-Day Travel Blueprint

### Trip Summary

### Financial Overview

### Day-by-Day Itinerary

### Essential Money-Saving Tips

### Things to Remember

### Packing & Practical Notes

### Final Recommendations
```

### Day-by-day requirements

Each day contains approximately 3–5 practical points.

The system avoids:

```text
Morning
Afternoon
Evening
```

subsections.

Instead, it uses chronological actions such as:

```text
Day 1: Travel to the destination

- Take the recommended train from Hyderabad to Tirupati.
- Transfer from the station to the hotel using practical local transport.
- Check in and keep the remaining time for nearby sightseeing.
```

The itinerary must clearly state actual travel actions instead of vague labels such as only `Transfer`, `Hotel`, or `Land at airport`.

---

# 👤 Human-in-the-Loop (HITL)

## 10. Human reviews the draft

The workflow does **not** immediately finalize the itinerary.

After the Itinerary Agent creates the draft, LangGraph reaches the `human_approval` node.

The workflow pauses using LangGraph's `interrupt()` mechanism and sends the draft to the frontend.

The user can choose:

```text
Approve & Generate Final
```

or:

```text
Revise Using Feedback
```

Example feedback:

```text
Reduce hotel spending and add one more sightseeing activity.
```

This is the Human-in-the-Loop checkpoint.

---

## 11. Approval or revision resumes the workflow

The frontend sends the human decision to:

```text
POST /api/travel/approve
```

with:

```json
{
  "thread_id": "...",
  "approved": true,
  "feedback": ""
}
```

or:

```json
{
  "thread_id": "...",
  "approved": false,
  "feedback": "Reduce accommodation cost and keep the sightseeing sequence."
}
```

LangGraph resumes the paused thread using the stored checkpoint state.

---

# 🧾 Final Response

## 12. Final Response Agent produces the polished plan

The Final Response Agent receives:

```text
Human approval / revision feedback
User request
Supervisor constraints
Flight results
Hotel results
Weather results
Budget analysis
Draft itinerary
```

It then generates the final TravelMind AI blueprint while preserving the required output structure.

The final response is returned to FastAPI and displayed on the website.

The user can then:

```text
Copy the plan
Download the plan as PDF
Start another request
```

---

# 💾 PostgreSQL + LangGraph Persistence

TravelMind AI uses PostgreSQL as the LangGraph checkpoint store.

The backend creates a PostgreSQL connection and passes it to:

```python
PostgresSaver
```

The compiled graph therefore maintains workflow state using a `thread_id`.

This is important for the HITL flow because the graph must remember the draft itinerary and resume the same workflow after the user approves or requests revisions.

Conceptually:

```text
User Request
    ↓
LangGraph Thread ID
    ↓
PostgreSQL Checkpoint
    ↓
Draft + State Saved
    ↓
Human Review
    ↓
Resume Same Thread
    ↓
Final Response
```

---

# 🔌 MCP Architecture

TravelMind AI uses the **Model Context Protocol (MCP)** to connect the planning agents with external tools.

## MCP integrations

### Tavily

Remote MCP endpoint:

```text
https://mcp.tavily.com/mcp/
```

Used primarily by the Hotel Agent for web research.

### AviationStack

Local stdio MCP command:

```text
uvx aviationstack-mcp
```

Used by the Flight Agent.

### Custom Weather MCP

Local stdio MCP server:

```text
custom_weather_mcp_server.py
```

Used for OpenWeather current weather and forecast data.

---

# 🧩 MCP Client Helpers

`mcp_client.py` centralizes the MCP integrations and exposes helper functions used by the LangGraph agents.

```text
tavily_mcp_search()
aviation_mcp_call()
weather_mcp_search()
forecast_mcp_search()
extract_destination()
```

The backend therefore does not need to manage each MCP implementation directly inside every agent.

---

# 🏗️ System Architecture

```text
                         ┌───────────────────────┐
                         │      Web Browser       │
                         │ HTML/CSS/JavaScript    │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │       FastAPI         │
                         │        app.py         │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │      LangGraph        │
                         │      backend.py       │
                         └───────────┬───────────┘
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
             ┌───────────────┐               ┌───────────────┐
             │   Guardrail   │               │   Supervisor  │
             └───────┬───────┘               └───────┬───────┘
                     │                               │
                     └───────────────┬───────────────┘
                                     ▼
                  ┌─────────────────────────────────────┐
                  │         Specialist Agents           │
                  │                                     │
                  │ Flight │ Hotel │ Weather │ Budget   │
                  └─────────────────┬───────────────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │ Itinerary Agent │
                           └────────┬────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │  HITL Approval  │
                           └────────┬────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │  Final Agent    │
                           └────────┬────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │ Final Blueprint │
                           └─────────────────┘

External services:

  Groq ───────────────► LLM inference
  Tavily MCP ─────────► Hotel/web research
  AviationStack MCP ──► Flight/airport/airline data
  OpenWeather MCP ────► Weather + forecast
  PostgreSQL ─────────► LangGraph checkpoints
```

---

# 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python** | Backend application and agent logic |
| **FastAPI** | REST API and web application server |
| **LangGraph** | Agent orchestration, routing, state, and HITL workflow |
| **LangChain** | LLM and message abstractions |
| **Groq** | LLM inference |
| **GPT-OSS 20B** | Current planning model: `openai/gpt-oss-20b` |
| **MCP** | Standardized tool integration |
| **Tavily** | Web/hotel research |
| **AviationStack** | Aviation data |
| **OpenWeather** | Weather and forecast data |
| **PostgreSQL** | Persistent LangGraph checkpoint storage |
| **Jinja2** | HTML templating |
| **HTML/CSS/JavaScript** | Frontend interface |
| **Docker** | Containerized deployment |
| **uv / uvx** | MCP command execution, including AviationStack MCP |

---

# 📁 Project Structure

```text
Multi_Agent_Travel_Planning_System/
│
├── app.py                         # FastAPI application and API routes
├── backend.py                     # LangGraph graph, agents, routing, HITL, checkpointing
├── mcp_client.py                  # MCP client and external tool helpers
├── custom_weather_mcp_server.py   # Custom OpenWeather MCP server
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Production container definition
├── .env                           # Local secrets/configuration (not committed)
├── .gitignore                     # Git exclusions
│
├── templates/
│   └── index.html                 # TravelMind AI web interface
│
├── static/
│   ├── style.css                  # Frontend styling
│   └── script.js                  # API, workflow, HITL and PDF interaction
│
├── tools/
│   └── ...                        # Supporting travel/search integrations
│
└── excalidraw_files/
    └── ...                        # Architecture/design diagrams
```

---

# 🔐 Environment Variables

Create `.env` in the project root for local development.

```env
DATABASE_URL=postgresql://user:password@host:5432/travel_db
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
AVIATION_STACK_API_KEY=your_aviationstack_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
DEFAULT_ORIGIN_IATA=HYD
```

### Important

Do **not** commit `.env` to GitHub.

The current project uses **Hyderabad (HYD)** as the default origin for travel requests where an origin is otherwise not specified.

---

# ✅ Prerequisites

Before running TravelMind AI locally, install:

- Python 3.10+
- PostgreSQL
- `uv` / `uvx`
- API credentials for Groq, Tavily, AviationStack, and OpenWeather

Verify `uvx` is available:

```bash
uvx --version
```

---

# 📦 Installation

## 1. Clone the repository

```bash
git clone <your-repository-url>
cd Multi_Agent_Travel_Planning_System
```

## 2. Create a virtual environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure `.env`

Add the required environment variables described above.

---

# ▶️ Run Locally

Start the application:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:8000/
```

Health check:

```text
http://127.0.0.1:8000/health
```

---

# 🌐 API Endpoints

## `GET /health`

Returns the application health/status.

Example:

```bash
curl http://127.0.0.1:8000/health
```

## `POST /api/travel`

Starts a new travel-planning workflow or continues using a supplied thread ID.

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/travel \
  -H "Content-Type: application/json" \
  -d '{"message":"Plan a 3-day trip to Tirupati from Hyderabad for 2 people under ₹10000 total."}'
```

A successful request can return:

```text
thread_id
answer
requires_approval
approval_request
flight_results
hotel_results
weather_results
budget_results
itinerary
selected_agents
trip_constraints
supervisor_reasoning
guardrail_allowed
llm_calls
```

## `POST /api/travel/approve`

Resumes the paused HITL workflow.

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/travel/approve \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"YOUR_THREAD_ID","approved":true,"feedback":""}'
```

For revision:

```bash
curl -X POST http://127.0.0.1:8000/api/travel/approve \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"YOUR_THREAD_ID","approved":false,"feedback":"Reduce accommodation cost and keep the itinerary practical."}'
```

---

# 🐳 Docker

The application includes a `Dockerfile` and is designed to run as a containerized FastAPI service.

Build:

```bash
docker build -t travelmind-ai .
```

Run locally:

```bash
docker run --env-file .env -p 8000:8000 travelmind-ai
```

Open:

```text
http://127.0.0.1:8000/
```

For production deployment, configure the environment variables in the hosting platform rather than committing secrets into the image or repository.

---

# ☁️ Deployment Architecture

A typical production deployment is:

```text
GitHub Repository
       ↓
   Render Build
       ↓
     Docker
       ↓
FastAPI / Uvicorn
       ↓
TravelMind AI
  ┌────┼───────────────────┐
  ↓    ↓       ↓           ↓
Groq Tavily AviationStack OpenWeather
  ↓
PostgreSQL / LangGraph Checkpoints
```

The container listens on port `8000`.

For Render or another container platform, provide the required environment variables through the platform's secret/environment configuration.

---

# 🛡️ Error Handling and Fallbacks

TravelMind AI includes several practical fallbacks.

### Guardrail failure

If guardrail processing fails, the backend can fall back to allowing the request rather than unnecessarily blocking valid travel planning.

### Supervisor parsing failure

If Supervisor JSON parsing fails, the workflow falls back to the complete specialist-agent sequence.

### Hotel MCP failure

The hotel agent falls back to general accommodation guidance and explicitly treats it as non-live advice.

### Weather MCP failure

The weather agent falls back to seasonal travel guidance and recommends verifying the actual forecast before departure.

### Unsupported live pricing

The system is instructed to distinguish estimates from confirmed live information and never claim a booking is confirmed unless supported by the supplied data.

---

# 💰 Travel Planning Rules

The planner follows several explicit rules.

## Budget

The user's stated budget is interpreted as the **total group budget** unless the user explicitly says otherwise.

## Accommodation

```text
1 room → 4 or fewer travelers
2 rooms → more than 4 travelers
```

Explicit accommodation requests override the default rule.

## India transport

The planner generally prefers:

```text
Train → Bus → Low-cost local transport
```

and only uses domestic flights when necessary, explicitly requested, or materially more practical.

## International transport

The plan starts with the international journey from Hyderabad and identifies connecting hubs when applicable, followed by practical local/intercity transportation at the destination.

## Itinerary

The requested duration is preserved, and every requested day should be represented in the final plan.

---

# 📄 Final Travel Output Format

The application is designed to produce the following output structure:

```markdown
# [Destination / Trip Name]

## [X]-Day Travel Blueprint

### Trip Summary

### Financial Overview

| Expense Category | Estimated Cost (INR) |
| --- | --- |
| Flights / Intercity Travel | ... |
| Accommodation | ... |
| Local Transport | ... |
| Food & Miscellaneous | ... |
| Activities / Sightseeing | ... |
| Total Projected Cost | ... |

### Day-by-Day Itinerary

Day 1: ...
- ...
- ...
- ...

Day 2: ...
- ...
- ...
- ...

### Essential Money-Saving Tips

### Things to Remember

### Packing & Practical Notes

### Final Recommendations
```

The frontend renders the Markdown result and can export it as an A4 PDF.

---

# 🔁 State Management

TravelMind AI maintains structured state across the workflow.

Important state fields include:

```text
user_query
guardrail_allowed
guardrail_reason
selected_agents
trip_constraints
supervisor_reasoning
flight_results
hotel_results
weather_results
budget_results
itinerary
approval_request
approved
human_feedback
final_response
llm_calls
```

This allows each specialist agent to contribute information without losing the context required by downstream stages.

---

# 🧪 Testing Strategy

A practical validation sequence is:

```text
1. Start the app
2. Open /health
3. Submit a small travel request
4. Verify Supervisor routing
5. Verify specialist MCP calls
6. Verify itinerary generation
7. Verify HITL draft appears
8. Approve or request revision
9. Verify final response
10. Test PDF export
```

For production testing, start with a smaller request before running a large multi-day trip because each workflow can require multiple LLM calls and external tool calls.

---

# ⚠️ API Rate Limits

TravelMind AI depends on third-party services and therefore is subject to their API limits.

In particular, Groq model requests can fail with HTTP `429` when the account reaches the applicable token or request limit.

A rate-limit error does **not** necessarily indicate that the TravelMind application itself is broken.

Similarly, external MCP/API failures should be distinguished from application logic failures when diagnosing production logs.

---

# 🔒 Security Notes

- Keep `.env` out of version control.
- Never expose API keys in frontend JavaScript.
- Store production secrets in the deployment platform's environment/secret manager.
- Do not treat estimated travel prices as confirmed bookings.
- Validate any live travel information before making real-world reservations.

---

# 🤝 Contributing

Contributions are welcome.

```text
1. Fork the repository
2. Create a feature branch
3. Make focused changes
4. Test locally
5. Commit your changes
6. Open a pull request
```

For architectural changes, preserve the existing LangGraph state flow and output format unless the change intentionally modifies the product behavior.

---

# 🙏 Acknowledgments

TravelMind AI combines several open-source and API technologies to demonstrate a practical agentic AI application:

- LangGraph
- LangChain
- FastAPI
- MCP
- Groq
- Tavily
- AviationStack
- OpenWeather
- PostgreSQL

The project is intended as a practical example of building a production-oriented, stateful, multi-agent application with external tools and human approval.

---

# 👨‍💻 Author

**Prem Kumar**

Built & designed by Prem Kumar ✦

---

# 📌 Project Summary

TravelMind AI demonstrates an end-to-end agentic workflow:

```text
User
 ↓
FastAPI
 ↓
Guardrail
 ↓
Supervisor
 ↓
Dynamic Specialist Agents
 ├── Flight + AviationStack MCP
 ├── Hotel + Tavily MCP
 ├── Weather + Custom OpenWeather MCP
 └── Budget Analysis
 ↓
Itinerary Agent
 ↓
Draft Blueprint
 ↓
Human-in-the-Loop
 ↓
Final Response Agent
 ↓
PostgreSQL Checkpointed State
 ↓
Travel Blueprint + PDF
```

The result is a single application that combines **LLM reasoning, multi-agent orchestration, real-world tool calls, persistent workflow state, human approval, and practical travel planning** in one deployable system.
