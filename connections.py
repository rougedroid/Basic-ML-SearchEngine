#neo4j connection
from neo4j import GraphDatabase
URI = "neo4j://127.0.0.1:7687"
USERNAME = "neo4j"
PASSWORD = "testpass"

# Establish connection to Neo4j
driver = GraphDatabase.driver(
    URI,
    auth=(USERNAME, PASSWORD)       
)   

answer_node_model_path = 'answer_node_model.pkl'
answer_depth_model_path = 'answer_depth_model.pkl'
answer_tone_model_path = 'answer_tone_model.pkl'