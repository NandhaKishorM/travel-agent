import os
import googlemaps
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain_google_community import GoogleSearchAPIWrapper
from pydantic_ai import RunContext
from agent.agent import agent

gmaps = googlemaps.Client(key=os.environ["GOOGLE_CLIENT_API_KEY"])
search_client = GoogleSearchAPIWrapper()


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