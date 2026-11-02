
# simple_agent.py

import random
from openai import OpenAI


class SimpleAgent:

    def __init__(
        self,
        model="nemotron-3-super:120b",
        api_key="sk-c2877b22d20342d299f0cf11bbd5f932",
        base_url="http://cs-steverw.umflint.edu:8080/api",
        history_length=8
    ):
        """
        Initialize the LLM Actor.

        Args:
            model: Model name served by Ollama.
            api_key: API key for the OpenAI-compatible client.
            base_url: API endpoint.
            history_length: Number of previous actions/observations
                            to include in the prompt.
        """

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        self.model = model
        self.history_length = history_length

        self.history = []

    # ---------------------------------------------------------
    # HISTORY
    # ---------------------------------------------------------

    def add_history(self, observation, action, location):
        """Record the latest interaction."""

        self.history.append({
            "observation": observation,
            "action": action,
            "location": location
        })

        # Retain only recent interactions.
        self.history = self.history[-self.history_length:]

    def format_history(self):
        """Format history for the LLM prompt."""

        if not self.history:
            return "No previous actions."

        formatted = []

        for step in self.history:
            formatted.append(
                f"Location: {step['location']}\n"
                f"Observation: {step['observation']}\n"
                f"Action: {step['action']}"
            )

        return "\n\n".join(formatted)

    # ---------------------------------------------------------
    # ACTION SELECTION
    # ---------------------------------------------------------

    def choose_action(
        self,
        observation,
        current_location,
        valid_actions,
        known_locations=None
    ):
        """
        Select the next action using the LLM.

        Args:
            observation: Current observation from Jericho.
            current_location: Current location name.
            valid_actions: Actions available in the environment.
            known_locations: Locations already present in the map.

        Returns:
            A valid action string.
        """

        if not valid_actions:
            return "look"

        valid_actions = list(valid_actions)

        # Convert action list into prompt-friendly text.
        actions_text = "\n".join(
            f"- {action}" for action in valid_actions
        )

        known_locations = known_locations or []

        locations_text = (
            ", ".join(known_locations)
            if known_locations
            else "No locations discovered yet."
        )

        history_text = self.format_history()

        system_prompt = """
You are an autonomous exploration agent playing Zork I.

Your primary objective is to explore the environment and discover
new locations and transitions.

You should:
- Find your way underground.
- Search for treasures and other useful items.
- Explore unfamiliar directions and areas.
- Use observations to understand your surroundings.
- Avoid repeating actions that recently failed.
- Prefer actions that may reveal new locations.
- Remember that some actions may require objects or puzzle solutions.
- Select exactly one action from the available action list.

Do not explain your reasoning.
Return only the exact action string.
"""

        user_prompt = f"""
CURRENT LOCATION:
{current_location}

CURRENT OBSERVATION:
{observation}

KNOWN LOCATIONS:
{locations_text}

RECENT HISTORY:
{history_text}

AVAILABLE ACTIONS:
{actions_text}

Choose the single best action to continue exploring.

Return only the exact action string from AVAILABLE ACTIONS.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                temperature=0.2,
                max_tokens=100,

                # Open WebUI workaround
                extra_body={
                    "parent_id": None
                }
            )

            action = response.choices[0].message.content.strip() # type: ignore

            # Validate action against Jericho's action list.
            if action in valid_actions:
                return action

            # Handle responses with quotes or whitespace.
            action = action.strip("\"' ")

            if action in valid_actions:
                return action

            # If LLM output is invalid, select a valid action.
            print(f"Invalid LLM action: {action}")
            print("Selecting a random valid action instead.")

            return random.choice(valid_actions)

        except Exception as e:
            print(f"LLM request failed: {e}")

            # Fall back to a valid action.
            return random.choice(valid_actions)
