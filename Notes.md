# Project: Thinker

### Description: 
This project aims to give ML models the ability to think about thoughts, a.k.a. metacognition along with persistent reliable storage in the form of a Neo4j nodal-graph database. In this primitive version, the river database will take a JSON input and output a JSON with information about the question the prompt asks. The River model will get its information from the graph database. 

### Tools:
1. RiverML
2. Python
3. Neo4j

## Thoughts:
* We can have the ml predict "node id" and this way, it can go to that node, read the json in that node, and follow the path upto "n" depth to get info or, follow the path to run instructions. But the problem with this method is that the ml itself doesn't know what it is, like it thinks its just a number...... and words would be better cuz then it would "know" but in hindsight its an ml model, no brain or soul, putting words instead of letters only changes what binary it sees, it doesn't extract meaning from it like chatgpt would extract meaning from words and relate them. 

* Okay so i want the core thinking to be multiple parts, a main one, and then an answering engine ig, and then a doer, etc etc. so the main one can be a deterministic script aswell atp. its only job is to see what the intent tag is, and then direct it towards the required model. And each of these smaller models are good at their one job. so like the mood predictor model will see all the mood info and decide tone and depth of response. and then the ranker model will search and rank all the nodes depending on the prompt, like if its a question, it'll search for fact nodes, if its a task, then it'll search for procedure nodes, etc. etc. once the nodes are searched and predicted, this will reach either the output llm in case of question to answer the question, or it will reach the worker model, the worker model goes to the nodes, and at every node/split in graph, it will decide which branch to follow based on context. 

* Right now i will build the ranker node to answer questions. 

* Okay built the answer core, and i learnt the hard way that i choose a wrong model for the task. I thought a FM just takes multiplication of the input data line in meaning, but it literally miltiplies them so can't have dictionaries in there. But this is a simple fix. Just change the ML model and run training again. The trainer script is ready, and i can make it test too with a single comment change so its fine. I did like 4-5 training  batches to remove the errors in my code cuz fucking vs code isn't doing intellisense, its just suggesting AI snippets.

* Upon testing the tree model, results were horrible, possibly becuase the tree only sees a given data once or twice, and it was text or smtn. I was getting equal probability for all nodes. Now, i decided that i'll use FMClassifier only because it works with grouped relationships etc. So, to fix the text being multiplied problem, i decided to vectorise the input values in preprocessing. BUt we need to build a better version which incorporates relationships, and has better vectorisation instead of just dumping everything in there and making the model figure it out. 

## Learning Notes:
* The kind of ML model is very important, do some reading on the different types available and find the best fit model for your issue, do not blindly take AI's opinion on this matter. 

* When data is less, train model on the data multiple times using epoch's this is to let it learn properly. Do remember, excessive training on one small set of data might lead to tunnel-vision for the AI and produce horrible results with new data. Also, we could use this in teaching our model how to run commands, and the entire procedure training, because we want a certain level of tunnel vision there, we don't want it to wander off into a separate chain of commands instead of following the current branch. 

### Model Options: 
Linear models
Tree models
Ensemble models
Naive Bayes
Nearest neighbors
Factorization machines
Neural-network-like models
Clustering
Anomaly detection
Recommendation/ranking
Time series
Multi-output/meta models

* Okay, in the architecture, we need multiple models to do different tasks, of these, we need a model to figure out the nearest neighbours and other shit. 

### ML Problem: 
* We are going to input a json/dictionary which will contain some values from a predefined list of factors. The model must use these values to determine which "node" it should refer to. This is closest to the "question" prompt in our actual model because it will be doing this exact thing, it will look at the search values and point the python to a node and give it some depth/breadth. the depth/breadth will depend on the llm inputs and the mood settings. 
* For this information model, a tree classifier appears to be the best right now. This finalises the architecture of the entire main project, we will need multiple small models which get called upon by:
1. Deterministic logic or
2. Another River ML that is trained on this * later

