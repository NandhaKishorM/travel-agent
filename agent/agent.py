from pathlib import Path
from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits
from typing import List, Dict
import os
import googlemaps
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain_google_community import GoogleSearchAPIWrapper
from tavily import TavilyClient


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MODEL = 'openai:gpt-4o'
gmaps = googlemaps.Client(key=os.environ["GOOGLE_CLIENT_API_KEY"])
search_client = GoogleSearchAPIWrapper()
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# Load system prompt for main agent
prompt_file = Path(__file__).parent / "prompt.txt"
with open(prompt_file, "r", encoding="utf-8") as file:
    SYSTEM_PROMPT = file.read()

# Main agent
agent = Agent(model=MODEL, system_prompt=SYSTEM_PROMPT, instrument=True)

# Tool functions for the main agent
@agent.tool
def get_places(ctx: RunContext, query: str, location: Optional[str] = None, radius: int = 5000) -> List[Dict[str, Any]]:
    """
    Search for places of interest using Google Maps Places API.
    
    Args:
        query: Search term (e.g., "restaurants", "museums")
        location: Optional location to center the search (latitude,longitude)
        radius: Search radius in meters (default 5000)
        
    Returns:
        List of places matching the search criteria
    """
    if location:
        lat, lng = map(float, location.split(","))
        return gmaps.places(query=query, location=(lat, lng), radius=radius).get('results', [])
    else:
        return gmaps.places(query=query).get('results', [])

@agent.tool
def geocode_address(ctx: RunContext, address: str) -> List[Dict[str, Any]]:
    """
    Geocode an address to get coordinates and location details.
    
    Args:
        address: Address to geocode
        
    Returns:
        Geocoding results with coordinates and address components
    """
    return gmaps.geocode(address)

@agent.tool
def reverse_geocode_coordinates(ctx: RunContext, latitude: float, longitude: float) -> List[Dict[str, Any]]:
    """
    Reverse geocode coordinates to get address information.
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        
    Returns:
        Address information for the coordinates
    """
    return gmaps.reverse_geocode((latitude, longitude))

@agent.tool
def get_directions(ctx: RunContext, origin: str, destination: str, mode: str = "driving") -> List[Dict[str, Any]]:
    """
    Get directions between two locations.
    
    Args:
        origin: Starting location
        destination: Destination location
        mode: Transportation mode (driving, walking, bicycling, transit)
        
    Returns:
        Directions with route information
    """
    return gmaps.directions(origin, destination, mode=mode, departure_time=datetime.now())

@agent.tool
def get_current_weather(ctx: RunContext, latitude: float, longitude: float) -> Dict[str, Any]:
    """
    Get current weather conditions for given coordinates.
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        
    Returns:
        Current weather data including temperature, conditions, etc.
    """
    params = {
        "key": os.environ["GOOGLE_API_KEY"],
        "location.latitude": latitude,
        "location.longitude": longitude,
    }
    
    url = "https://weather.googleapis.com/v1/currentConditions:lookup"
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Weather API error: {response.status_code}"}
    
@agent.tool
def get_weather_forecast(ctx: RunContext, latitude: float, longitude: float, days: int = 7) -> Dict[str, Any]:
    """
    Get weather forecast for the next few days.
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        days: Number of days to forecast (default 7)
        
    Returns:
        Weather forecast data
    """
    params = {
        "key": os.environ["GOOGLE_API_KEY"],
        "location.latitude": latitude,
        "location.longitude": longitude,
        "days": days,
    }
    
    url = "https://weather.googleapis.com/v1/forecast/days:lookup"
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Weather API error: {response.status_code}"}
    
@agent.tool
def get_current_location(ctx: RunContext) -> Dict[str, Any]:
    """
    Get the current location of the user based on IP address.
    
    Returns:
        Current location data including city, region, country, and coordinates
    """
    try:
        response = requests.get("https://ipinfo.io/json")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": f"Could not get current location: {str(e)}"}
    
@agent.tool
def get_current_date_time(ctx: RunContext) -> str:
    """
    Get the current date and time in ISO format.
    
    Returns:
        Current date and time as a string
    """
    return datetime.now().isoformat()

@agent.tool
def search_web(ctx: RunContext, query: str, num_results: int = 5) -> List[Dict[str, str]]:
    response = tavily_client.search(query)
    return response

@agent.tool
def validate_address(ctx: RunContext, addresses: List[str], region_code: str = 'US') -> Dict[str, Any]:
    """
    Validate addresses using Google Maps Address Validation API.
    
    Args:
        addresses: List of addresses to validate
        region_code: Region code (default: 'US')
        
    Returns:
        Address validation results
    """
    try:
        return gmaps.addressvalidation(addresses, regionCode=region_code)
    except Exception as e:
        return {"error": f"Address validation error: {str(e)}"}

# Hotel specialist agent
agent_hotels = Agent(
    model=MODEL,
    system_prompt="Analyze the travel response context and extract location information. Use the 'find_hotels' tool to search for hotels in those locations. Always return hotel search results.",
    instrument=True
)

@agent_hotels.tool
def find_hotels(ctx: RunContext, query: str, num_results: int = 5) -> List[Dict[str, str]]:
    query = f"hotels in {query}"
    response = tavily_client.search(query)
    return response

# Hotel reviews specialist agent
agent_reviews = Agent(
    model=MODEL,
    system_prompt="Analyze hotel search results and get detailed reviews for the hotels. Use the 'get_hotel_reviews' tool to search for reviews of the hotels found from multiple platforms. Focus on gathering comprehensive review information from TripAdvisor, Google Reviews, and social media platforms.",
    instrument=True
)

@agent_reviews.tool
def get_hotel_reviews(ctx: RunContext, hotel_name: str, location: str) -> Dict[str, Any]:
    """
    Search for reviews of a specific hotel across multiple platforms.
    
    Args:
        hotel_name: Name of the hotel
        location: Location of the hotel
        
    Returns:
        Hotel reviews and rating information from multiple platforms
    """
    platforms = [
        f"{hotel_name} {location} site:tripadvisor.com reviews",
        f"{hotel_name} {location} site:google.com reviews", 
        f"{hotel_name} {location} site:instagram.com",
        f"{hotel_name} {location} site:twitter.com OR site:x.com",
        f"{hotel_name} {location} site:linkedin.com",
        f"{hotel_name} {location} site:facebook.com reviews"
    ]
    
    platform_results = {}
    platform_names = ["tripadvisor", "google", "instagram", "twitter", "linkedin", "facebook"]
    
    for i, platform_query in enumerate(platforms):
        platform_name = platform_names[i]
        try:
            response = tavily_client.search(platform_query, max_results=2)
            if response and isinstance(response, dict) and 'results' in response:
                platform_results[platform_name] = response['results']
            elif response and isinstance(response, list):
                platform_results[platform_name] = response
            else:
                platform_results[platform_name] = []
        except Exception as e:
            platform_results[platform_name] = [{"error": f"Failed to search {platform_query}: {str(e)}"}]
    
    return {
        "hotel_name": hotel_name,
        "location": location,
        "reviews_by_platform": platform_results,
        "total_platforms_searched": len(platforms)
    }

# Final travel plan agent
agent_planner = Agent(
    model=MODEL,
    system_prompt="You are a travel planning specialist. Analyze the main travel response, hotel search results, and hotel reviews to create a comprehensive final travel plan. Combine all information into a well-structured, actionable travel itinerary.",
    instrument=True
)

@agent_planner.tool
def create_final_plan(ctx: RunContext, main_response: str, hotel_data: str, review_data: str) -> Dict[str, str]:
    """
    Create a final comprehensive travel plan.
    
    Args:
        main_response: Main travel agent response
        hotel_data: Hotel search results
        review_data: Hotel reviews data
        
    Returns:
        Comprehensive travel plan combining all information
    """
    return {
        "plan_type": "comprehensive_travel_plan",
        "main_info": main_response,
        "hotels": hotel_data,
        "reviews": review_data,
        "status": "plan_created"
    }

# Agent registry for dynamic detection
AGENT_REGISTRY = {
    "agent_hotels": {
        "agent": agent_hotels,
        "tools": ["find_hotels"],
        "emoji": "🏨",
        "description": "Hotel specialist agent"
    },
    "agent_reviews": {
        "agent": agent_reviews,
        "tools": ["get_hotel_reviews"],
        "emoji": "⭐",
        "description": "Hotel reviews specialist agent"
    },
    "agent_planner": {
        "agent": agent_planner,
        "tools": ["create_final_plan"],
        "emoji": "📋",
        "description": "Final travel plan agent"
    },
    "master_coordinator": {
        "agent": agent,
        "tools": ["get_places", "geocode_address", "reverse_geocode_coordinates", "get_directions", 
                 "get_current_weather", "get_weather_forecast", "get_current_location", 
                 "get_current_date_time", "search_web", "validate_address"],
        "emoji": "🤖",
        "description": "Master coordinator agent"
    }
}

def get_agent_for_tool(tool_name: str) -> tuple[str, str]:
    """
    Dynamically determine which agent handles a specific tool.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Tuple of (agent_name, agent_emoji)
    """
    for agent_name, agent_info in AGENT_REGISTRY.items():
        if tool_name in agent_info["tools"]:
            return agent_name, agent_info["emoji"]
    
    # Default to master coordinator
    return "master_coordinator", AGENT_REGISTRY["master_coordinator"]["emoji"]

class TravelAgent:
    def __init__(self):
        self.agent = agent
        self.agent_hotels = agent_hotels
        self.agent_reviews = agent_reviews
        self.agent_planner = agent_planner

    def run(self, query: str, model: str = MODEL):
        # Get main agent response
        response = self.agent.run_sync(query, model=model)
        main_response = response.output
        
        # Check if hotels are mentioned using simple string matching
        if "hotel" in main_response.lower() or "stay" in main_response.lower() or "accommodation" in main_response.lower():
            # Use main response as context for hotel agent
            hotel_response = self.agent_hotels.run_sync(
                f"Based on this travel context: {main_response}\n\nFind hotels for the locations mentioned.",
                model=model
            )
            
            # Get hotel reviews based on hotel search results
            review_response = self.agent_reviews.run_sync(
                f"Based on these hotel search results: {hotel_response.output}\n\nGet detailed reviews for these hotels from multiple platforms including TripAdvisor, Google Reviews, Instagram, Twitter/X, LinkedIn, and Facebook.",
                model=model
            )
            
            # Create final comprehensive travel plan
            final_plan_response = self.agent_planner.run_sync(
                f"Create a comprehensive travel plan using:\n\nMain Response: {main_response}\n\nHotel Data: {hotel_response.output}\n\nReview Data: {review_response.output}",
                model=model
            )
            
            # Combine traces from all agents
            import json
            
            main_trace = response.all_messages_json()
            hotel_trace = hotel_response.all_messages_json()
            review_trace = review_response.all_messages_json()
            plan_trace = final_plan_response.all_messages_json()
            
            # Parse JSON traces to lists
            traces = [main_trace, hotel_trace, review_trace, plan_trace]
            combined_trace = []
            
            for trace in traces:
                if isinstance(trace, bytes):
                    trace = json.loads(trace.decode('utf-8'))
                elif isinstance(trace, str):
                    trace = json.loads(trace)
                combined_trace.extend(trace)
            
            return final_plan_response.output, combined_trace
        
        return main_response, response.all_messages_json()
