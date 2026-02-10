"""Generate a 1000-turn synthetic conversation for evaluation."""

import json
import random
from pathlib import Path


# Pre-written filler messages — varied topics, no memorable personal info
FILLER_MESSAGES = [
    "What's the weather like today?",
    "Tell me a fun fact about octopuses.",
    "How do I make pasta from scratch?",
    "What's the capital of Mongolia?",
    "Can you explain how WiFi works?",
    "What's a good movie to watch tonight?",
    "How do I fix a leaky faucet?",
    "Tell me about the history of chess.",
    "What's the difference between HTTP and HTTPS?",
    "How do plants photosynthesize?",
    "What's happening in the world of cricket?",
    "How do I bake chocolate chip cookies?",
    "Explain quantum computing simply.",
    "What's a good book recommendation?",
    "How does GPS work?",
    "Tell me about the solar system.",
    "What are some tips for better sleep?",
    "How do I learn a new language effectively?",
    "What's the tallest building in the world?",
    "Can you write a short poem about rain?",
    "How does a car engine work?",
    "What are the benefits of meditation?",
    "Tell me about the Amazon rainforest.",
    "How do airplanes fly?",
    "What's a good exercise routine for beginners?",
    "How do I organize my desk better?",
    "What's the most spoken language in the world?",
    "How do vaccines work?",
    "Tell me about ancient Egyptian civilization.",
    "What's the best way to save money?",
    "How do magnets work?",
    "What makes the sky blue?",
    "How do I improve my writing skills?",
    "What's the deepest point in the ocean?",
    "How do computers store data?",
    "What's a good strategy game?",
    "How do I take care of indoor plants?",
    "Tell me about the theory of relativity.",
    "What's the fastest animal on earth?",
    "How do I make a good cup of coffee?",
    "What is machine learning?",
    "How do bridges support so much weight?",
    "Tell me about the Renaissance period.",
    "What's a good recipe for soup?",
    "How do smartphones work?",
    "What causes earthquakes?",
    "How do I start a garden?",
    "What's the difference between a virus and bacteria?",
    "How do electric cars work?",
    "Tell me about space exploration.",
]


def generate_conversation(output_path: str = "eval/conversation_1000.json"):
    """Generate a 1000-turn conversation mixing planted memories and filler."""
    
    # Load scenarios
    with open("eval/scenarios.json") as f:
        scenarios = json.load(f)
    
    plants = {p["turn"]: p for p in scenarios["plants"]}
    probes = {p["turn"]: p for p in scenarios["probes"]}
    
    conversation = []
    
    for turn_id in range(1, 1001):
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
    
    Path(output_path).parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(conversation, f, indent=2)
    
    plant_count = len(plants)
    probe_count = len(probes)
    filler_count = 1000 - plant_count - probe_count
    print(f"Generated 1000-turn conversation:")
    print(f"  Plants: {plant_count}")
    print(f"  Probes: {probe_count}")
    print(f"  Filler: {filler_count}")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    generate_conversation()
