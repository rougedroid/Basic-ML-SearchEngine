import pickle 
import json
import neo4j
from river import facto
from river import tree
import connections
from neo4j import GraphDatabase

def init_model():
    try: 
        with open(connections.answer_depth_model_path, "rb") as f:
            depth_model = pickle.load(f)
        
    except:
        print("Model loading failed.")
        print("Given Path:", connections.answer_depth_model_path  )
        
        
        depth_model = (
            preprocessing.StandardScaler() |
            multioutput.RegressorChain(tree.HoeffdingTreeRegressor())  # Initialize an empty model if loading fails
        )
    try: 
        with open(connections.answer_tone_model_path, "rb") as f:
            tone_model = pickle.load(f)
        
    except:
        print("Model loading failed.")
        print("Given Path:", connections.answer_tone_model_path  )
        
        tone_model = (
            preprocessing.StandardScaler() |
            tree.HoeffdingTreeClassifier()  # Initialize an empty model if loading fails
        )
    return depth_model, tone_model

def get_answer_depth(input_dict, model):
    # Parse input JSON
    data = input_dict
    
    # Extract features for the model
    features = {
        "intent": data.get("intent", ""),
        "keyword": data.get("keyword", ""),
        "context": data.get("context", ""),
        "additional_info": data.get("additional_info", ""),
        "phrase": data.get("phrase", ""),
        "mood": data.get("mood", "")
    }
    
    # Predict node_id using the model
    predicted_depth = model.predict_one(features)
    
    return predicted_depth

def get_answer_tone(input_dict, model):
    # Parse input JSON
    data = input_dict
    
    # Extract features for the model
    features = {
        "intent": data.get("intent", ""),
        "keyword": data.get("keyword", ""),
        "context": data.get("context", ""),
        "additional_info": data.get("additional_info", ""),
        "phrase": data.get("phrase", ""),
        "mood": data.get("mood", "")
    }
    
    # Predict node_id using the model
    predicted_tone = model.predict_one(features)
    
    return predicted_tone

def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)

def get_answer_features(input_dict):
    # Parse input JSON
    depth_model, tone_model = init_model()
    data = input_dict
    
    if input_dict.get("correct_depth") is not None:
        depth_model.learn_one({
            "intent": data.get("intent", ""),
            "keyword": data.get("keyword", ""),
            "context": data.get("context", ""),
            "additional_info": data.get("additional_info", ""),
            "phrase": data.get("phrase", ""),
            "mood": data.get("mood", "")
        }, input_dict["correct_depth"])
    
    
    if input_dict.get("correct_tone") is not None:
        tone_model.learn_one({
            "intent": data.get("intent", ""),
            "keyword": data.get("keyword", ""),
            "context": data.get("context", ""),
            "additional_info": data.get("additional_info", ""),
            "phrase": data.get("phrase", ""),
            "mood": data.get("mood", "")
        }, input_dict["correct_tone"])

    depth = [get_answer_depth(data, depth_model)]
    tone = get_answer_tone(data, tone_model)
    
    save_model(depth_model, connections.answer_depth_model_path)
    save_model(tone_model, connections.answer_tone_model_path)   
    
    
    return depth,tone   



