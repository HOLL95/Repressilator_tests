"""
Main entry point for running the Repressilator analysis as a module.

Usage:
    python -m repressilator_analysis
"""

from .pipeline import run_analysis

if __name__ == "__main__":
    run_analysis()
