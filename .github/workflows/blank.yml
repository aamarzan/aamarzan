import numpy as np

def suggest_model(dataset_info):
    """
    Suggests a deep learning model based on dataset characteristics.
    
    Parameters:
    dataset_info (dict): A dictionary containing dataset details.
    
    Returns:
    dict: Suggested model type and architecture details.
    """
    problem_type = dataset_info.get("problem_type", "").lower()
    input_shape = dataset_info.get("input_shape", None)
    num_classes = dataset_info.get("num_classes", None)
    data_type = dataset_info.get("data_type", "").lower()
    
    model_suggestion = {"model": None, "architecture": None}

    if problem_type in ["classification", "multi-class classification"]:
        if data_type == "image":
            model_suggestion["model"] = "Convolutional Neural Network (CNN)"
            model_suggestion["architecture"] = {
                "conv_layers": 3,
                "filters": [32, 64, 128],
                "kernel_size": (3, 3),
                "pooling": "MaxPooling",
                "dense_layers": 2,
                "units": [128, num_classes],
                "activation": "softmax"
            }
        elif data_type == "text":
            model_suggestion["model"] = "Recurrent Neural Network (RNN) or Transformer"
            model_suggestion["architecture"] = {
                "embedding_dim": 128,
                "rnn_type": "LSTM",
                "lstm_units": 256,
                "dense_units": num_classes,
                "activation": "softmax"
            }
        elif data_type == "sequence":
            model_suggestion["model"] = "Long Short-Term Memory (LSTM)"
            model_suggestion["architecture"] = {
                "lstm_layers": 2,
                "lstm_units": [128, 64],
                "dropout": 0.2,
                "dense_units": num_classes,
                "activation": "softmax"
            }
        else:
            model_suggestion["model"] = "Multi-Layer Perceptron (MLP)"
            model_suggestion["architecture"] = {
                "hidden_layers": 3,
                "units": [128, 64, 32],
                "activation": "relu",
                "output_activation": "softmax"
            }

    elif problem_type in ["regression"]:
        if data_type == "tabular":
            model_suggestion["model"] = "Feedforward Neural Network (MLP)"
            model_suggestion["architecture"] = {
                "hidden_layers": 3,
                "units": [128, 64, 32],
                "activation": "relu",
                "output_units": 1,
                "output_activation": "linear"
            }
        elif data_type == "time-series":
            model_suggestion["model"] = "LSTM for Time-Series Regression"
            model_suggestion["architecture"] = {
                "lstm_layers": 2,
                "lstm_units": [128, 64],
                "dropout": 0.2,
                "output_units": 1,
                "output_activation": "linear"
            }
        else:
            model_suggestion["model"] = "MLP"
            model_suggestion["architecture"] = {
                "hidden_layers": 2,
                "units": [64, 32],
                "activation": "relu",
                "output_activation": "linear"
            }

    elif problem_type in ["object detection", "segmentation"]:
        model_suggestion["model"] = "Convolutional Neural Network (CNN) - ResNet/Faster R-CNN"
        model_suggestion["architecture"] = {
            "pretrained_model": "ResNet50/Faster R-CNN",
            "fine_tuning": True,
            "layers_trainable": "Last few layers"
        }
    
    else:
        model_suggestion["model"] = "Unknown dataset type. Please provide more details."

    return model_suggestion


# Example usage
dataset_info = {
    "problem_type": "classification",
    "data_type": "image",
    "input_shape": (224, 224, 3),
    "num_classes": 10
}

best_model = suggest_model(dataset_info)
print("Best Model Suggestion:", best_model)
