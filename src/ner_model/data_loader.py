from datasets import load_dataset, DatasetDict

def load_data(dataset_name: str) -> DatasetDict:
    """
    Loads a dataset from the Hugging Face Hub.

    Args:
        dataset_name (str): The name of the dataset on the Hugging Face Hub.

    Returns:
        DatasetDict: The loaded dataset.
    """
    print(f"Loading dataset '{dataset_name}'...")
    return load_dataset(dataset_name)

def get_label_maps():
    """
    Returns the label list, label2id, and id2label mappings for the NER task.

    Returns:
        tuple: A tuple containing (label_list, label2id, id2label).
    """
    label_list = [
        "O", "B-人名", "I-人名", "B-法人名", "I-法人名", "B-政治的組織名", "I-政治的組織名",
        "B-その他の組織名", "I-その他の組織名", "B-地名", "I-地名", "B-施設名", "I-施設名",
        "B-製品名", "I-製品名", "B-イベント名", "I-イベント名"
    ]
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    return label_list, label2id, id2label
