import torch
import os
import json
import argparse
from torchvision import transforms
from PIL import Image
from networks import ResNetSE  # Import the SE-enhanced model
from collections import Counter
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

'''
############################################################################################################################################################################
Run the following command in the terminal to train the SE-Enhanced ResNet model:

python test_squeeze_model.py --model_path ResNetModels\squeeze_Resnet_epoch10_model.pth --image_dir data\SoccerNet\jersey-2023\processed_data\test\common_data\crops\imgs --output_file predictions.json --illegible_file data\SoccerNet\jersey-2023\processed_data\test\common_data\illegible_results.json --ground_truth_file data\SoccerNet\jersey-2023\extracted\test\test_gt.json
#############################################################################################################################################################################
'''
# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(model_path, device):
    model = ResNetSE()  # Initialize the SE-enhanced model
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load model weights
    model.to(device)
    model.eval() 
    return model

def predict(model, image_path, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(image)  # Forward pass to get the predictions

    pred_label = torch.argmax(out, dim=1).item()  # Get the predicted label

    return pred_label

def batch_inference(model, image_dir, output_file, device, illegible_file):
    results = {}
    illegible_ids = set()

    # Load illegible data
    with open(illegible_file, 'r') as f:
        illegible_data = json.load(f)
        illegible_ids = set(illegible_data.get("illegible", []))

    # Dictionary to count predictions for each tracklet_id
    tracklet_predictions = {}

    for image_name in os.listdir(image_dir):
        if image_name.endswith(('.jpg', '.png')):
            image_path = os.path.join(image_dir, image_name)
            pred_label = predict(model, image_path, device)  # Get the predicted label

            # Extract tracklet ID from filename (before the first underscore)
            tracklet_id = image_name.split('_')[0]

            # If tracklet_id is already in the dictionary, append the prediction
            if tracklet_id not in tracklet_predictions:
                tracklet_predictions[tracklet_id] = []
            tracklet_predictions[tracklet_id].append(pred_label)

    # Aggregate most common label for each tracklet_id
    for tracklet_id, predictions in tracklet_predictions.items():
        most_common_label = Counter(predictions).most_common(1)[0][0]
        results[tracklet_id] = str(most_common_label)  # Store as string

    # Add illegible IDs to the results with label -1
    for illegible_id in illegible_ids:
        if illegible_id not in results:  # Ensure we don't overwrite existing predictions
            results[illegible_id] = "-1"

    # Save predictions to JSON
    os.makedirs("SqueezeResNetResults", exist_ok=True)
    output_file_path = os.path.join("SqueezeResNetResults", output_file)
    
    with open(output_file_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Predictions saved to {output_file_path}")

    # Create final_squeeze_results by adding illegible results
    final_results = results.copy()

    # Save the final results as final_squeeze_results.json
    final_output_file = os.path.join("SqueezeResNetResults", "final_squeeze_results.json")
    with open(final_output_file, 'w') as f:
        json.dump(final_results, f, indent=4)

    print(f"Final predictions saved to {final_output_file}")

    return final_results

# Evaluation functions
def accuracy(y_true, y_pred):
    return np.mean(np.array(y_true) == np.array(y_pred)) * 100

def precision_recall_f1(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return precision, recall, f1

def mean_average_precision(y_true, y_pred, n_classes):
    mAP = 0.0
    for i in range(n_classes):
        y_true_class = [1 if label == i else 0 for label in y_true]
        y_pred_class = [1 if label == i else 0 for label in y_pred]
        if sum(y_true_class) > 0:
            mAP += average_precision_score(y_true_class, y_pred_class)

    return mAP / n_classes if n_classes > 0 else 0.0

# Main evaluation
def evaluate_model(results_file, ground_truth_file):
    with open(results_file, 'r') as f:
        results = json.load(f)

    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)

    y_pred = []
    y_true = []
    
    for tracklet_id in ground_truth:
        if tracklet_id in results:
            y_pred.append(int(results[tracklet_id]))
            y_true.append(int(ground_truth[tracklet_id]))

    # Accuracy
    acc = accuracy(y_true, y_pred)
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)
    mAP = mean_average_precision(y_true, y_pred, n_classes=10)  # Assuming 10 classes (digits 0-9)

    print(f"Accuracy: {acc}%")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")
    print(f"Mean Average Precision (mAP): {mAP}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory of cropped images')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save predictions')
    parser.add_argument('--illegible_file', type=str, required=True, help='Path to illegible.json file')
    parser.add_argument('--ground_truth_file', type=str, required=True, help='Path to ground truth JSON file')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, device)  # Load the model
    final_results = batch_inference(model, args.image_dir, args.output_file, device, args.illegible_file)  # Run inference

    # Evaluate the model using the final predictions and ground truth
    evaluate_model("SqueezeResNetResults/final_squeeze_results.json", args.ground_truth_file)
