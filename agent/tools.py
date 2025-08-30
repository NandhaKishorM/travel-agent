# NOTE: This file is no longer needed with the updated agent.py
# All tools are now registered directly in agent.py using @agent.tool_plain decorators
# This file is kept for reference only

import os
import googlemaps
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain_google_community import GoogleSearchAPIWrapper
from pydantic_ai import RunContext, Agent

# This approach is DEPRECATED in favor of direct tool registration in agent.py
# The new approach prevents tool name conflicts and follows latest pydantic_ai patterns

def register_tools(agent: Agent):
    """
    DEPRECATED: Tool registration function.
    
    This approach can cause tool name conflicts. Use the new approach in agent.py instead
    where tools are registered directly using @agent.tool_plain decorators.
    """
    print("WARNING: register_tools() is deprecated. Tools are now registered directly in agent.py")
    print("This function call will be ignored to prevent tool conflicts.")
    
    # Do nothing to prevent conflicts with the new decorator-based approach
    pass

# Legacy tool functions - these are now implemented as decorators in agent.py
# Keeping for reference only

gmaps = googlemaps.Client(key=os.environ.get("GOOGLE_CLIENT_API_KEY", ""))
search_client = GoogleSearchAPIWrapper()

def legacy_get_places(ctx: RunContext, query: str, location: Optional[str] = None, radius: int = 5000) -> List[Dict[str, Any]]:
    """
    LEGACY: Search for places of interest using Google Maps Places API.
    This function is replaced by find_places() in agent.py
    """
    if location:
        lat, lng = map(float, location.split(","))
        return gmaps.places(query=query, location=(lat, lng), radius=radius).get('results', [])
    else:
        return gmaps.places(query=query).get('results', [])

def legacy_geocode_address(ctx: RunContext, address: str) -> List[Dict[str, Any]]:
    """
    LEGACY: Geocode an address to get coordinates and location details.
    This function is replaced by geocode_address() in agent.py
    """
    return gmaps.geocode(address)

def legacy_reverse_geocode_coordinates(ctx: RunContext, latitude: float, longitude: float) -> List[Dict[str, Any]]:
    """
    LEGACY: Reverse geocode coordinates to get address information.
    This function is replaced by reverse_geocode_coordinates() in agent.py
    """
    return gmaps.reverse_geocode((latitude, longitude))

def legacy_get_directions(ctx: RunContext, origin: str, destination: str, mode: str = "driving") -> List[Dict[str, Any]]:
    """
    LEGACY: Get directions between two locations.
    This function is replaced by get_directions() in agent.py
    """
    return gmaps.directions(origin, destination, mode=mode, departure_time=datetime.now())

def legacy_get_current_weather(ctx: RunContext, latitude: float, longitude: float) -> Dict[str, Any]:
    """
    LEGACY: Get current weather conditions for given coordinates.
    This function is replaced by get_current_weather() in agent.py
    """
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

def legacy_get_weather_forecast(ctx: RunContext, latitude: float, longitude: float, days: int = 7) -> Dict[str, Any]:
    """
    LEGACY: Get weather forecast for the next few days.
    This function is replaced by get_weather_forecast() in agent.py
    """
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

def legacy_get_current_location(ctx: RunContext) -> Dict[str, Any]:
    """
    LEGACY: Get the current location of the user based on IP address.
    This function is replaced by get_current_location() in agent.py
    """
    try:
        response = requests.get("https://ipinfo.io/json")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": f"Could not get current location: {str(e)}"}

def legacy_get_current_date_time(ctx: RunContext) -> str:
    """
    LEGACY: Get the current date and time in ISO format.
    This function is replaced by get_current_date_time() in agent.py
    """
    return datetime.now().isoformat()

def legacy_search_web(ctx: RunContext, query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    FIXED: Search the web using the stable DDGS library. This avoids the SSLError.
    """
    from ddgs import DDGS
    try:
        with DDGS() as ddgs:
            results = [
                {"title": r.get('title'), "href": r.get('href'), "body": r.get('body')}
                for r in ddgs.text(query, max_results=num_results)
            ]
        return results
    except Exception as e:
        return [{"error": f"Web search failed: {str(e)}"}]
@agent.tool
def validate_address(ctx: RunContext, addresses: List[str], region_code: str = 'US') -> Dict[str, Any]:
    """
    LEGACY: Validate addresses using Google Maps Address Validation API.
    This function is replaced by validate_address() in agent.py
    """
    try:
        return gmaps.addressvalidation(addresses, regionCode=region_code)
    except Exception as e:
        return {"error": f"Address validation error: {str(e)}"}