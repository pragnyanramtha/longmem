
import json
import random
import os

# Pre-written filler messages 
FILLER_MESSAGES = [
    "What's the weather like today?",
    "Tell me a fun fact about octopuses.",
    "How do I make pasta from scratch?",
    "What's the capital of Mongolia?",
    "Can you explain how WiFi works?",
    "What's a good movie to watch tonight?",
]

def generate_quick_conversation():
    output_path = "eval/conversation_quick.json"
    scenario_path = "eval/scenarios_quick.json"
    
    with open(scenario_path) as f:
        scenarios = json.load(f)
    
    plants = {p["turn"]: p for p in scenarios["plants"]}
    probes = {p["turn"]: p for p in scenarios["probes"]}
    
    conversation = []
    
    # Generate 50 turns
    for turn_id in range(1, 51):
        if turn_id in plants:
            msg = plants[turn_id]["content"]
            msg_type = "plant"
        elif turn_id in probes:
            msg = probes[turn_id]["content"]
            msg_type = "probe"
        else:
            msg = random.choice(FILLER_MESSAGES)
            msg_type = "filler"
        
        conversation.append({
            "turn_id": turn_id,
            "role": "user",
            "content": msg,
            "type": msg_type,
        })
    
    with open(output_path, "w") as f:
        json.dump(conversation, f, indent=2)
    
    print(f"Generated 50-turn conversation to {output_path}")

if __name__ == "__main__":
    generate_quick_conversation()
