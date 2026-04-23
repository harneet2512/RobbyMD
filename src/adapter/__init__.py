"""Adapter layer between substrate state and downstream readers.

Each adapter takes a question (and a post-ingestion substrate connection)
and produces the prompt inputs for a reader. Replaces ad-hoc retrieval +
reader wiring that previously lived in eval harnesses.
"""
