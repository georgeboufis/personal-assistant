"""
state.py
--------
Ορίζει το "σχήμα" των δεδομένων που κουβαλάει ο agent από βήμα σε βήμα
(node σε node) μέσα στο LangGraph.

Γιατί το κάνουμε έτσι:
- Το LangGraph χρειάζεται ένα TypedDict (ή pydantic model) που περιγράφει
  το state, ώστε να ξέρει πώς να το ενημερώνει σε κάθε node.
- Το `messages` είναι λίστα από LangChain message objects (HumanMessage,
  AIMessage, ToolMessage, κλπ) - είναι το "ιστορικό" της συνομιλίας.
- Το `add_messages` είναι ένας ειδικός "reducer": αντί κάθε node να πρέπει
  να ξαναγράφει ΟΛΗ τη λίστα μηνυμάτων, απλά επιστρέφει τα ΝΕΑ μηνύματα
  και το LangGraph τα προσθέτει αυτόματα στο τέλος της υπάρχουσας λίστας.
  Χωρίς αυτό, θα έπρεπε σε κάθε node να κάνουμε χειροκίνητα
  `state["messages"] + [new_message]`, κάτι που είναι εύκολο να ξεχαστεί
  και να χαθεί ιστορικό.
"""

from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Το state του agent.

    Προς το παρόν κρατάμε μόνο το ιστορικό μηνυμάτων. Αργότερα, όταν
    προσθέσουμε tools (calendar, gmail), πιθανόν να προσθέσουμε κι άλλα
    πεδία εδώ (π.χ. `user_id`, `pending_tool_calls`), αλλά η δομή θα
    παραμείνει: TypedDict + Annotated reducers όπου χρειάζεται.
    """

    messages: Annotated[list[BaseMessage], add_messages]
