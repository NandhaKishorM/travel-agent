import os
import googlemaps
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from pydantic_ai import Agent, RunContext
from langchain_google_community import GoogleSearchAPIWrapper

# Read system prompt
prompt_file = Path(__file__).parent / "prompt.txt"
with open(prompt_file, "r", encoding="utf-8") as file:
    SYSTEM_PROMPT = file.read()

# Initialize Google Maps client
gmaps = googlemaps.Client(key=os.environ.get("GOOGLE_CLIENT_API_KEY", ""))
search_client = GoogleSearchAPIWrapper()

# Create agent with updated pydantic_ai patterns
agent = Agent(
    'openai:gpt-4o',  # Default model
    system_prompt=SYSTEM_PROMPT,
    retries=2
)

# --- Tool Registration using Latest pydantic_ai Patterns ---

@agent.tool_plain
def geocode_address(address: str) -> List[Dict[str, Any]]:
    """
    Geocode an address to get coordinates and location details.
    
    Args:
        address: Address to geocode
        
    Returns:
        Geocoding results with coordinates and address components
    """
    try:
        return gmaps.geocode(address)
    except Exception as e:
        return [{"error": f"Geocoding failed: {str(e)}"}]

@agent.tool_plain
def reverse_geocode_coordinates(latitude: float, longitude: float) -> List[Dict[str, Any]]:
    """
    Reverse geocode coordinates to get address information.
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        
    Returns:
        Address information for the coordinates
    """
    try:
        return gmaps.reverse_geocode((latitude, longitude))
    except Exception as e:
        return [{"error": f"Reverse geocoding failed: {str(e)}"}]

@agent.tool_plain
def get_directions(origin: str, destination: str, mode: str = "driving") -> List[Dict[str, Any]]:
    """
    Get directions between two locations.
    
    Args:
        origin: Starting location
        destination: Destination location
        mode: Transportation mode (driving, walking, bicycling, transit)
        
    Returns:
        Directions with route information
    """
    try:
        return gmaps.directions(origin, destination, mode=mode, departure_time=datetime.now())
    except Exception as e:
        return [{"error": f"Directions failed: {str(e)}"}]

@agent.tool_plain
def get_current_weather(latitude: float, longitude: float) -> Dict[str, Any]:
    """
    Get current weather conditions for given coordinates.
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        
    Returns:
        Current weather data including temperature, conditions, etc.
    """
    try:
        params = {
            "key": os.environ.get("GOOGLE_API_KEY", ""),
            "location.latitude": latitude,
            "location.longitude": longitude,
        }
        
        url = "https://weather.googleapis.com/v1/currentConditions:lookup"
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Weather API error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Weather request failed: {str(e)}"}

@agent.tool_plain
def get_weather_forecast(latitude: float, longitude: float, days: int = 7) -> Dict[str, Any]:
    """
    Get weather forecast for the next few days.
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        days: Number of days to forecast (default 7)
        
    Returns:
        Weather forecast data
    """
    try:
        params = {
            "key": os.environ.get("GOOGLE_API_KEY", ""),
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
    except Exception as e:
        return {"error": f"Weather forecast failed: {str(e)}"}

@agent.tool_plain
def get_current_location() -> Dict[str, Any]:
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

@agent.tool_plain
def get_current_date_time() -> str:
    """
    Get the current date and time in ISO format.
    
    Returns:
        Current date and time as a string
    """
    return datetime.now().isoformat()

@agent.tool_plain
def search_web(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Search the web using Google Custom Search.
    
    Args:
        query: Search query
        num_results: Number of results to return (default 5)
        
    Returns:
        List of search results with title, link, and snippet
    """
    try:
        return search_client.results(query, num_results=num_results)
    except Exception as e:
        return [{"error": f"Web search failed: {str(e)}"}]

@agent.tool_plain
def validate_address(addresses: List[str], region_code: str = 'US') -> Dict[str, Any]:
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

@agent.tool_plain
def find_places(query: str, location: Optional[str] = None, radius: int = 5000) -> List[Dict[str, Any]]:
    """
    Search for places of interest using Google Maps Places API.
    
    Args:
        query: Search term (e.g., "restaurants", "museums")
        location: Optional location to center the search (latitude,longitude)
        radius: Search radius in meters (default 5000)
        
    Returns:
        List of places matching the search criteria
    """
    try:
        if location:
            lat, lng = map(float, location.split(","))
            return gmaps.places(query=query, location=(lat, lng), radius=radius).get('results', [])
        else:
            return gmaps.places(query=query).get('results', [])
    except Exception as e:
        return [{"error": f"Places search failed: {str(e)}"}]

class TravelAgent:
    def __init__(self):
        self.agent = agent

    def run(self, query: str, model: str = 'openai:gpt-4o'):
        """
        Run the travel agent with the given query.
        
        Args:
            query: User's travel query
            model: Model to use for the agent
            
        Returns:
            Tuple of (response, trace_messages)
        """
        try:
            response = self.agent.run_sync(query, model=model)
            return response.output, response.all_messages_json()
        except Exception as e:
            error_msg = f"Agent execution failed: {str(e)}"
            return error_msg, []