# Make consideration for different kind of relations etc etc...

import pickle 
import json
import neo4j
from river import facto
from neo4j import GraphDatabase
from river import tree
import connections
from river import feature_extraction
from river import multioutput
from river import preprocessing
from google import genai
import nodepointer
import time


bow = feature_extraction.BagOfWords()
client = genai.Client()
gem_model = "gemini-3.1-flash-lite" # "gemini-pro" --- IGNORE ---

"""
def init_model(path):
    try: 
        with open(path, "rb") as f:
            model = pickle.load(f)
    except:
        print("Model loading failed.")
        print("Given Path:", path  )

        
        model = (
            preprocessing.StandardScaler() |
            facto.FMClassifier()  # Initialize an empty model if loading fails
        )
    return model
"""


def search_nodes(keyword):
    # This function will search for nodes in the Neo4j database that match the keyword
    with connections.driver.session(database='cskg') as session:
        result = session.run(
            "MATCH (n) WHERE n.name CONTAINS $keyword RETURN n",
            keyword=keyword
        )
        nodes = [record["n"] for record in result]
    return nodes

def neighbour_nodes(node_id):
    # This function will return the neighboring nodes of a given node_id
    with connections.driver.session(database='cskg') as session:
        result = session.run(
            "MATCH (n)-[]->(m) WHERE n.id = $node_id RETURN m",
            node_id=node_id
        )
        neighbors = [record["m"]["name"] for record in result]
    return neighbors

def next_node_data(input_dict):
    
    keyword = input_dict["keyword"]
    candidate_nodes = search_nodes(keyword)
    predictions = []
    for candidate in candidate_nodes:
        #candidate_properties = candidate.items()  # Get properties of the candidate node
        #print("Candidate Node:", candidate)
        x = {
            "keyword": keyword,
            "context": input_dict["context"],
            "additional_info": input_dict["additional_info"],
            "phrase": input_dict["phrase"],
            #"mood": input_dict["mood"],
            "neighbours": bow.transform_one(str(neighbour_nodes(candidate.id))),  # Add neighboring nodes as features
            # Add candidate properties to the feature set
            **candidate,
            "candidate_id": candidate.id
        }
        predictions.append((candidate.id, x))
    
    # Select the candidate node with the highest prediction score
    return predictions
    


def main():
    seed = ""
    """
    Have a prompt for google gemini. 1. to generate the seed prompt. then, use seed prompt to manually get candidate node list. then use the candidate node list data and the seed prompt to generate the solution to next node. then, use this node id to train the model. then, repeat the process until the depth is reached for given seed prompt.

    keywords to use: 
    {
    intent: "question", #fixed
    keyword: "quantum computing", #variable
    context: "general", #variable - can be general or specific
    additional_info: "potential applications", #variable
    phrase: "what is keyword and additional_info", #variable ( include actual question by user without keyword and additional info to train model on general english phrases)
    mood: "curious" #variable ( can be curious, neutral, or negative )
    
    
    # correction: "false"
    # correct node id: "4:dafe80c1-d719-4025-9d56-73d5bf016bbc:5"
    }
    """

    seeder_prompt = """ You are training a node pointer model that predicts the next node in a reasoning process based on the current input. The input includes an intent, a keyword, context, additional information, a phrase, and a mood. The model should use this information to predict the most relevant next node in the reasoning process. Your task is to generate a seed prompt that can be used to train the model. The seed prompt should include a sample input and the expected output (the next node id). The input should be in the form of a JSON object with the following structure:
{
    "intent": "question",
    "keyword": "quantum computing",
    "context": "general",
    "additional_info": "potential applications",
    "phrase": "what is keyword and additional_info",
    "mood": "curious"
}   

here, the intent is fixed as "question", but the rest of the fields can vary. for the phrase field, include actual question by user without keyword and additional info to train model on general english phrases. The mood field can be "curious", "neutral", or "negative".

ONLY OUTPUT THE RAW JSON FORMAT. DO NOT INCLUDE ANY EXPLANATION OR ADDITIONAL TEXT OR JSON CODE BLOCK FORMATTING. THE PROVIDED JSON IS THE FINAL OUTPUT FORMAT. YOUR OUTPUS SHOULD START WITH { and END WITH } AND DIRECTLY GET INTO THE REQUIRED FIELDS. DO NOT SAY "the output is" or "the json is" or anything like that. JUST OUTPUT THE RAW JSON.
"""
    
    
    for i in range(30):
        print("Iteration:", i+1)

        time.sleep(5)  # Add a delay to avoid hitting rate limits or overwhelming the model with requests
        
        response = client.models.generate_content(
            model=gem_model,
            contents=seeder_prompt
        )
        #print("Generated Seed Prompt:", response.text)
        response_dict = {
    "input": {
        "intent": "question",
        "keyword": "renewable energy",
        "context": "sustainability",
        "additional_info": "impact on climate change",
        "phrase": "how does this affect our future environment",
        "mood": "curious"
    },
    "next_node_id": "node_742"
}
        seed = json.loads(response.text)["input"]
        #seed = response_dict["input"]
        seed["correction"] = "false"
        #node_prediction = nodepointer.next_node_predictor(seed, connections.answer_node_model_path)
        predict_next_node = f""" 
        Based on the json provided below, and the candidate nodes provided, predict the next node in the reasoning process. The input JSON is:
        {seed}

        The candidate nodes are:
        {next_node_data(seed)}
        return the correct candidate node id as output.
        RETURN ONLY THE NODE ID AS RAW OUTPUT. DO NOT INCLUDE ANY EXPLANATION OR ADDITIONAL TEXT. YOUR OUTPUT SHOULD BE THE NODE ID STRING/NUMBER ONLY, WITHOUT ANY QUOTES OR CODE BLOCKS. ABSOLUTELY NO ADDITIONAL TEXT OR FORMATTING, JUST THE RAW NODE ID.
        """


        #seed["previous_node_id"] = node_prediction

        
        
        response = client.models.generate_content(
            model=gem_model,
            contents=predict_next_node
        )
        #print("Correct next node id:", response.text)

        train_input = seed
        train_input["correction"] = "true"
        train_input["correct_node_id"] = response.text
        print("Training model with input:", train_input)
        pred_id = nodepointer.next_node_predictor(train_input, connections.answer_node_model_path)
        #print("Predicted next node id:", pred_id)
        
def test():
    seed = {'intent': 'question', 'keyword': 'space', 'context': 'travel', 'additional_info': 'timeline', 'phrase': 'how does this affect our future', 'mood': 'curious', 'correction': 'false', 'correct_node_id': 'NULL'}
    print(nodepointer.next_node_predictor(seed, connections.answer_node_model_path))

    # we need richness in training data, rn gemini is teaching only quantum, or renewable energy, we need it to explore a large range of topics so the model learns the patterns of relations not the node patterns. 

if __name__ == "__main__":
    #main()
    test()




