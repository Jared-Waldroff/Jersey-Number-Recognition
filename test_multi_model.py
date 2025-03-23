import torch
import os
import json
import argparse
from torchvision import transforms
from PIL import Image
from networks import JerseyNumberMulticlassClassifier
from collections import Counter
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(model_path, device):
    model = JerseyNumberMulticlassClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(model, image_path, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        out1, out2, out3 = model(image)

    pred_label = torch.argmax(out1, dim=1).item()
    pred_digit1 = torch.argmax(out2, dim=1).item()
    pred_digit2 = torch.argmax(out3, dim=1).item()

    return pred_label, pred_digit1, pred_digit2

def batch_inference(model, image_dir, output_file, illegible_file, device):
    results = {}
    illegible = set()

    # Load illegible data from illegible.json
    if os.path.exists(illegible_file):
        with open(illegible_file, 'r') as f:
            illegible_data = json.load(f)
            illegible = set(illegible_data)

    for image_name in os.listdir(image_dir):
        if image_name.endswith(('.jpg', '.png')):
            image_path = os.path.join(image_dir, image_name)
            pred_label, pred_digit1, pred_digit2 = predict(model, image_path, device)

            # Extract tracklet ID from filename (remove extension)
            tracklet_id = os.path.splitext(image_name)[0]

            # If the tracklet is in the illegible list, mark it as -1
            if tracklet_id in illegible:
                results[tracklet_id] = "-1"
            else:
                if pred_digit2 == 10:  # Assuming 10 means "no second digit"
                    jersey_number = str(pred_digit1)
                else:
                    jersey_number = f"{pred_digit1}{pred_digit2}"

                results[tracklet_id] = jersey_number

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save results as JSON
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Predictions saved to {output_file}")

    return results

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

def evaluate_model(results, ground_truth_file):
    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)

    y_pred = []
    y_true = []
    
    for tracklet_id in ground_truth:
        if tracklet_id in results and results[tracklet_id] != "-1":
            y_pred.append(int(results[tracklet_id]))
            y_true.append(int(ground_truth[tracklet_id]))

    if not y_true:
        print("No valid predictions for evaluation.")
        return

    acc = accuracy(y_true, y_pred)
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)
    mAP = mean_average_precision(y_true, y_pred, n_classes=10)

    print(f"Evaluation Metrics:")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Mean Average Precision (mAP): {mAP:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory of cropped images')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save predictions')
    parser.add_argument('--illegible_file', type=str, required=True, help='Path to illegible.json file')
    parser.add_argument('--ground_truth_file', type=str, required=True, help='Path to ground truth JSON file')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, device)
    
    # Ensure the output file path includes the BaseLineMultiResNetResults directory
    output_file = os.path.join("BaseLineMultiResNetResults", args.output_file)

    results = batch_inference(model, args.image_dir, output_file, args.illegible_file, device)

    # Evaluate the model
    evaluate_model(results, args.ground_truth_file)
