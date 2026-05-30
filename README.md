# Basic Reasoning Engine

> **This project is abandoned and unfit for use.**
>
> It was an experimental prototype and should not be relied on for production or active development.

## Overview

This repository contains an early prototype for a hybrid ML/LLM personal assistant architecture. The goal was to identify the best node in a Neo4j graph to answer queries by combining machine learning and large language model reasoning.

The intended workflow was:
- store facts, procedures, and relationships in a Neo4j knowledge graph
- use feature engineering to score candidate graph nodes
- apply a hybrid ML retrieval method to surface the most relevant node

## Lessons Learned

After significant effort to make this method work, the author concluded that:
- basic topographical search is mostly sufficient for this type of retrieval
- a better approach would likely combine topological search with more advanced search algorithms such as Googles ML search engine methods

In other words, this repository represents a learning and prototype module rather than a finished system.

## Project Context

This code was meant to be part of a larger project called **Project Stark**. It served as a research and prototype stage for finding facts and procedures in a graph-based personal assistant system.

## Status

- abandoned
- not maintained
- unsuitable for production or any use ( previous versions would work for some cases after further training )

## Contents

- `answer.py` - was meant to be the main script called to answer questions, currently disfunctional
- `connections.py` - graph/database connection utilities
- `nodepointer.py` - node selection and pointer logic
- `trainer.py` - training utilities for the ML component
- `tonepredictor.py` - auxiliary prediction code
- `training_data.csv` - sample data for training and experimentation

## License

This repository is licensed under the MIT License. See `LICENSE` for details.
