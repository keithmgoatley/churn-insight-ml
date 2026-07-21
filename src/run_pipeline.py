"""Run the whole pipeline: generate data, train models, produce LLM summary."""

from generate_data import generate
from train_models import main as train
from llm_insights import summarise

if __name__ == "__main__":
    print("1/3 Generating data...")
    generate()
    print("2/3 Training models...")
    train()
    print("3/3 Producing LLM summary...")
    summarise()
    print("\nDone. See outputs/ for charts, metrics, and the executive summary.")
