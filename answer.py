"""
Answer Generation Module
This module is responsible for generating answers based on the input data and the trained models. It uses the node pointer model to predict the next node in the reasoning process, the depth predictor model to predict the depth of the answer, and the tone predictor model to predict the tone of the answer. The generated answer is then returned as a response to the user query.

Basically, main.py will send the input data to this program selectively. Once it comes here, this program will use the trained models to predict the next node, depth, and tone of the answer. 

This model will reitterate multiple times ( until depth is met ) and then, it will return the final node chain to main.py as facts, and main.py will then forward this to llm to give a natural language output to the user.

Process Steps:
1. Receive input data from main.py (including user query, context, etc.)
2. Use tonepredictor.py to predict the tone and depth of the answer based on the input data.
3. Use nodepointer.py to predict the next node in the reasoning process based on the input data.
4. Repeat step 3 until the predicted depth is reached.
5. Return the final node chain and tone details to main.py as facts for natural language generation.

"""

import connections
import nodepointer
import tonepredictor
import pickle
from neo4j import GraphDatabase



def get_tone_and_depth(input_dict):
    depth, tone = tonepredictor.get_answer_features(input_dict)
    return depth, tone

def get_next_node(input_dict):
    next_node_id = nodepointer.next_node_predictor(input_dict, connections.answer_node_model_path)
    return next_node_id

def get_answer(input_dict):
    #depth, tone = get_tone_and_depth(input_dict)
    input_dict["predicted_depth"] = 5 #depth
    input_dict["predicted_tone"] = 'good?' #tone
    
    node_chain = []
    for i in range(input_dict["predicted_depth"]):
        next_node_id = get_next_node(input_dict)
        node_chain.append(next_node_id)
        # Update input_dict with the new node information for the next iteration
        input_dict["previous_node_id"] = next_node_id
    
    return node_chain, tone
