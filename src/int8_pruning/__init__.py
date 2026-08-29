"""Shared infrastructure for the prune -> int8 pipeline, and its optional backends.

Stage-level code that is genuinely family-agnostic lives here. Anything that
knows about a specific model family belongs in `families/<name>/` instead.
"""
