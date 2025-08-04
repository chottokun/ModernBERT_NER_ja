from transformers import AutoModelForTokenClassification

def load_model(model_name: str, num_labels: int):
    """
    Loads an AutoModelForTokenClassification from a pretrained model name.

    Args:
        model_name (str): The name of the pretrained model.
        num_labels (int): The number of labels for the classification head.

    Returns:
        A transformers model object.
    """
    print(f"Loading model '{model_name}'...")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    return model
