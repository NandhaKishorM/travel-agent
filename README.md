# Travel Agent

A travel planning agent powered by OpenAI and Pydantic AI that helps you plan your trips and answer travel-related questions.

## Installation

### Install from Git

```bash
pip install git+https://github.com/yourusername/travel-agent.git
```

### Install for Development

```bash
git clone https://github.com/yourusername/travel-agent.git
cd travel-agent
pip install -e .
```

## Setup

### Environment Variables

Create a `.env` file in the project root directory with your API keys:

```bash
cp .env.example .env
```

Then edit the `.env` file with your actual API keys:

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your-openai-api-key-here

# Google API Configuration (optional)
GOOGLE_CLIENT_API_KEY=your-google-client-api-key-here
GOOGLE_CSE_ID=your-google-cse-id-here
GOOGLE_API_KEY=your-google-api-key-here
```


## Usage

### Command Line Interface

```bash
python run_agent.py --query "what is my current location" --model openai:gpt-4o --env .env
```

### Basic Usage

```python
from agent import TravelAgent
from load_dotenv import load_dotenv


# Create a travel agent instance
load_dotenv()  # Load environment variables from .env file
agent = TravelAgent()

# Ask a travel question
response, trace = agent.run("Plan a 3-day trip to Paris")
print(response)
```

### Example Queries

- "Plan a 5-day trip to Tokyo for a family of 4"
- "What's the best time to visit Greece?"
- "Suggest budget-friendly accommodations in Barcelona"
- "Create an itinerary for a weekend in New York City"

## Features

- AI-powered travel planning
- Comprehensive trip recommendations
- Message tracing for debugging
- Synchronous API for easy integration

## Requirements

- Python 3.8+
- OpenAI API key
- Internet connection

## Dependencies

- `openai>=1.0.0`
- `pydantic-ai`
- `python-dotenv`