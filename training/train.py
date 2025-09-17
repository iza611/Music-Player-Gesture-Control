from loguru import logger
import typer
from typing import Callable as function
from experiments import Experiment, experiments
from config import MODELS_DIR, KEYPOINT_NORM_DATA_DIR

app = typer.Typer()

EXPERIMENT_CONFIG = experiments["ridge_default_rocket_default"]

class MLPipeline:
    def __init__(self, steps: list[tuple[str, function]] = []):
        self.steps = steps

    def __str__(self):
        step_names = [name for name, _ in self.steps]
        return " -> ".join(step_names)

    def run(self, context):
        for name, func in self.steps:
            logger.info(f"Running step: {name}")
            context = func(context)
        return context

# Example step functions
def load_data(context):
    # ... load data ...
    context['dataloader'] = "dataloader"
    return context

def rocket_transform(context):
    # ... apply ROCKET transform ...
    context['transformed_data'] = "transformed_data"
    return context

def fit_classifier(context):
    # ... train model ...
    context['model'] = "trained_model"
    return context

def evaluate_model(context):
    # ... evaluate model ...
    context['evaluation'] = "evaluation_results"
    return context

def save_model(context):
    # ... save model ...
    context['model_path'] = MODELS_DIR / "model_name.pkl"
    return context

@app.command()
def main(
   # features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    # model_path: Path = MODELS_DIR / "model.pkl",
):
    pipeline = MLPipeline(steps=[
        ("load_data", load_data),
        # ("build_model", build_model),
        ("rocket_transform", rocket_transform),
        ("fit_classifier", fit_classifier),
        ("evaluate_model", evaluate_model),
        ("save_model", save_model),
    ])
    print(pipeline)

    context = pipeline.run(context={})
    print(context)
    logger.success("Modeling training complete.")

if __name__ == "__main__":
    app()
