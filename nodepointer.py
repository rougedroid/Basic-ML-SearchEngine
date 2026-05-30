# take a JSON input and return the node id to go to. 
"""
Input Format - user: 
{
    intent: "question",
    keyword: "quantum computing",
    context: "general",
    additional_info: "potential applications",
    phrase: "what is keyword and additional_info",
    mood: "curious"
    # correction: "false"
    # correct node id: "4:dafe80c1-d719-4025-9d56-73d5bf016bbc:5"
}
Have a correction parameter in json and strip it before sending it to the model. For training, we can have a separate correction json that has the same format as the input but with an additional field for the correct node id. And we can use that to train the model.

Input Format - Program:
{
    intent: "probe",
    keyword: "quantum computing",
    depth: 2,
    last_node: "4:dafe80c1-d719-4025-9d56-73d5bf016bbc:5"
    # correction: "false"
    # correct node id: "4:dafe80c1-d719-4025-9d56-73d5bf016bbc:5"
}
Output Format:
{
    node_id: "4:dafe80c1-d719-4025-9d56-73d5bf016bbc:5"
}

Model type: River ML Logistic Regression Classifier -> get probability

Database Action: Search for keyword in the node properties, and return all nodes that have the keyword. 

Model Action: Use the model to predict which node to go to next based on the input and the node properties. Return the node id of the predicted node.

"""

# Make consideration for different kind of relations etc etc...

import pickle 
import json
import neo4j
from neo4j import GraphDatabase
from river import facto
from river import tree
import connections
from river import feature_extraction
from river import linear_model
from river import multioutput
from river import preprocessing
from river import compose

bow = feature_extraction.BagOfWords()

# 1. Define how to handle text dynamically using a function wrapper
def preprocess_features(x):
    """
    Converts mixed graph/entity data into sparse numeric features
    suitable for:
        - LogisticRegression
        - FactorizationMachineClassifier
        - Online learning in River
    """

    

    bow = feature_extraction.BagOfWords(
        lowercase=True
    )

    features = {}

    # fields that should NEVER become numeric features
    banned_keys = {
        "id",
        "node_id",
        "edge_id",
        "uuid",
        "timestamp",
    }

    for key, value in x.items():

        # skip useless identifiers
        if key.lower() in banned_keys:
            continue

        # TEXT FEATURES
        if isinstance(value, str):

            tokens = bow.process_text(value)

            # binary sparse features
            for token in tokens:
                feature_name = f"{key}__{token}"
                features[feature_name] = 1.0

        # BOOLEAN FEATURES
        elif isinstance(value, bool):

            features[key] = float(value)

        # NUMERIC FEATURES
        elif isinstance(value, (int, float)):

            # avoid absurd magnitudes
            if abs(value) < 1e12:
                features[key] = float(value)

    return features


def init_model(path):
    try: 
        with open(path, "rb") as f:
            model = pickle.load(f)
        
    except:
        print("Model loading failed.")
        print("Given Path:", path  )

        
        model = (compose.FuncTransformer(preprocess_features) |facto.FMClassifier())  # Initialize an empty model if loading fails

    return model

def search_nodes(keyword):
    # This function will search for nodes in the Neo4j database that match the keyword
    with connections.driver.session(database='cskg') as session:
        result = session.run(
            "MATCH (n) WHERE n.name CONTAINS $keyword RETURN n",
            keyword=keyword
        )
        result = session.run(
            """
// 1. Search the full-text index for your keywords

CALL db.index.fulltext.queryNodes("nodeTextIndex", $searchKeyword) 

YIELD node, score AS textScore


// 2. Filter or ensure the GDS topology score parameter exists

WHERE node.topologyScore IS NOT NULL


// 3. Calculate a combined composite score 

// (Multiplying prevents low-text-matching nodes with massive topology scores from hijacking the results)

WITH node, textScore, (textScore * node.topologyScore) AS compositeScore


// 4. Sort by the highest composite score and limit to N results

ORDER BY compositeScore DESC

LIMIT 15


// 5. Return the top nodes and metrics

RETURN node



""", searchKeyword=keyword
        )
        print(result)
        nodes = [record["node"] for record in result]
    return nodes

def neighbour_nodes(node_id):
    # This function will return the neighboring nodes of a given node_id
    with connections.driver.session(database='cskg') as session:
        result = session.run(
            "MATCH (n)-[]->(m) WHERE elementId(n) = $node_id RETURN m",
            node_id=node_id
        )
        neighbors = [record["m"]["name"] for record in result]
    return neighbors

def predict_next_node(input_dict, model):
    
    # --- DIAGNOSTIC CHECK ---
    print("--- MODEL DIAGNOSTICS ---")
    print("Model Object Type:", type(model))
    
    # Check if it's a pipeline and inspect its inner classifier
    if hasattr(model, 'steps'):
        print("Pipeline steps discovered:", list(model.steps.keys()))
        classifier = model.steps.get('classifier') or list(model.steps.values())[-1]
        print("Underlying Classifier:", type(classifier))
        print("Underlying Classes:", getattr(classifier, 'classes', 'No classes attribute found'))
    else:
        print("Model Classes (Direct):", getattr(model, 'classes', 'No classes attribute found'))
    print("-------------------------")

    keyword = input_dict["keyword"]
    candidate_nodes = search_nodes(keyword)
    print("Retreived Nodes:", len(candidate_nodes))
    predictions = []
    print("Candidate Nodes:", [candidate.id for candidate in candidate_nodes])
    for candidate in candidate_nodes:
        #candidate_properties = candidate.items()  # Get properties of the candidate node
        x = {
            "keyword": keyword,
            "context": input_dict["context"],
            "additional_info": input_dict["additional_info"],
            "phrase": input_dict["phrase"],
            #"mood": input_dict["mood"],
            "neighbours": str(neighbour_nodes(candidate.id)),  # Add neighboring nodes as features
            # Add candidate properties to the feature set
            **dict(candidate),
            #"candidate_id": candidate.id
        }
        
        prediction = model.predict_proba_one(x)  # Predict the next node based on the input and candidate properties
        predictions.append((candidate.id, prediction[1.0]))  # Store the prediction for each candidate node
        print("Candidate Node ID:", candidate.id, "Prediction:", prediction)
    # Select the candidate node with the highest prediction score
    print("All Predictions:", predictions)
    prediction_probabilities = [pred[1] for pred in predictions]
    next_node_id = predictions[prediction_probabilities.index(max(prediction_probabilities))][0]
    return next_node_id

def train_model(input_dict, correct_node_id, model):
    keyword = input_dict["keyword"]
    candidate_nodes = search_nodes(keyword)
    for candidate in candidate_nodes:
        #print(dict(candidate))
        x = {
            "keyword": keyword,
            "context": input_dict["context"],
            "additional_info": input_dict["additional_info"],
            "phrase": input_dict["phrase"],
            #"mood": input_dict["mood"],
            "neighbours": str(neighbour_nodes(candidate.id)),  # Add neighboring nodes as features
            # Add candidate properties to the feature set
            **dict(candidate),
            #"candidate_id": candidate.id
        }
        #print(x)

        print("Training on candidate node ID:", candidate.id)
        y = 1.0 if candidate.id == int(correct_node_id) else 0.0  # Label the correct node as 1 and others as 0
        model.learn_one(x, y)  # Train the model incrementally with the new data

def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)

def next_node_predictor(input_dict, path):
    model = init_model(path)
    

    if input_dict["correction"] == "true":
        correct_node_id = input_dict["correct_node_id"]
        input_dict.pop("correction", None)  # Remove correction field before training
        input_dict.pop("correct_node_id", None)  # Remove correct_node_id field before
        train_model(input_dict, correct_node_id, model)
        # Use the correct node id to train the model
        # This part can be implemented based on how you want to structure your training data and process
        # For example, you can create a training example with the input features and the correct node id as the label
        # Then you can use model.learn_one(x, correct_node_id) to train the model incrementally
    else: 
        input_dict.pop("correction", None)  # Remove correction field if it exists  
        input_dict.pop("correct_node_id", None)  # Remove correct_node_id field if it exists
        next_node_id = predict_next_node(input_dict, model)
        save_model(model, path)
        return {"node_id": next_node_id}

    save_model(model, path)  # Save the model after training
    



