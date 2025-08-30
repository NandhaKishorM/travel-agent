from agent import TravelAgent

# Create a travel agent instance
agent = TravelAgent()

# Ask a travel question
response, trace = agent.run("Plan a 3-day trip to Paris")
print(response)