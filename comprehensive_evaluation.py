import torch
import os
import json
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from collections import Counter
from tqdm import tqdm
import scipy.stats as stats
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    accuracy_score,
    confusion_matrix, 
    average_precision_score
)

from networks import JerseyNumberClassifier, FPNResNet34, ResNetSE

'''
############################################################################################################################################################################
Comprehensive evaluation script that evaluates the jersey number recognition pipeline using various metrics:

1. Legibility Classification Evaluation:
   - Accuracy: Percentage of correctly classified legible and illegible images.
   - Precision: Proportion of true legible images among those predicted as legible.
   - Recall: Proportion of legible images correctly identified by the model.
   - F1-Score: Harmonic mean of precision and recall.

2. Jersey Number Recognition Evaluation:
   - Accuracy: Percentage of correctly recognized jersey numbers.
   - Digit-Level Accuracy: Accuracy of recognizing individual digits in multi-digit numbers.
   - Confidence Score: Average confidence score of the model's predictions.
   - Mean Average Precision (mAP): Average precision across all classes.

3. Overall Pipeline Evaluation:
   - End-to-End Accuracy: Percentage of correctly recognized jersey numbers across the entire pipeline.
   - Inference Time: Average time taken to process an image through the entire pipeline.
   - Robustness: Evaluate the model's performance under challenging conditions.

4. Statistical Significance:
   - Paired t-test: Perform a paired t-test to compare the performance of the models.

5. Efficiency Metrics:
   - Computational Cost: Measure the computational resources required.
   - Parameter Efficiency: Compare the number of parameters in the models.

Usage: python comprehensive_evaluation.py --model_paths model1.pth model2.pth model3.pth --model_names baseline fpn squeeze --image_dir path/to/images --illegible_file path/to/illegible.json --ground_truth_file path/to/gt.json --output_dir evaluation_results
#############################################################################################################################################################################
'''

# Global preprocessing transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Model loading functions
def load_model(model_path, model_type, device):
    """Load a model based on its type."""
    if model_type == 'baseline':
        model = JerseyNumberClassifier()
    elif model_type == 'fpn':
        model = FPNResNet34()
    elif model_type == 'squeeze':
        model = ResNetSE()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def predict_with_confidence(model, image_path, device):
    """Make a prediction and return both the predicted label and confidence score."""
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    start_time = time.time()
    with torch.no_grad():
        out = model(image_tensor)  # Forward pass
        probabilities = torch.nn.functional.softmax(out, dim=1)
    end_time = time.time()
    
    pred_label = torch.argmax(out, dim=1).item()  # Get the predicted label
    confidence = probabilities[0][pred_label].item()  # Get the confidence score
    
    return pred_label, confidence, end_time - start_time

def batch_inference(model, image_dir, illegible_file, device):
    """Run batch inference on all images in the directory."""
    results = {}
    confidences = {}
    inference_times = []
    illegible_ids = set()
    
    # Load illegible data
    with open(illegible_file, 'r') as f:
        illegible_data = json.load(f)
        illegible_ids = set(illegible_data.get("illegible", []))
    
    # Dictionary to count predictions for each tracklet_id
    tracklet_predictions = {}
    tracklet_confidences = {}
    
    image_list = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    
    for image_name in tqdm(image_list, desc="Processing Images", unit="image"):
        image_path = os.path.join(image_dir, image_name)
        pred_label, confidence, inference_time = predict_with_confidence(model, image_path, device)
        
        inference_times.append(inference_time)
        
        # Extract tracklet ID from filename (before the first underscore)
        tracklet_id = image_name.split('_')[0]
        
        # If tracklet_id is already in the dictionary, append the prediction and confidence
        if tracklet_id not in tracklet_predictions:
            tracklet_predictions[tracklet_id] = []
            tracklet_confidences[tracklet_id] = []
        
        tracklet_predictions[tracklet_id].append(pred_label)
        tracklet_confidences[tracklet_id].append(confidence)
    
    # Aggregate most common label for each tracklet_id
    for tracklet_id, predictions in tracklet_predictions.items():
        most_common_label = Counter(predictions).most_common(1)[0][0]
        results[tracklet_id] = str(most_common_label)  # Store as string
        
        # Get the confidences for all instances of the most common label
        common_label_indices = [i for i, p in enumerate(predictions) if p == most_common_label]
        common_label_confidences = [tracklet_confidences[tracklet_id][i] for i in common_label_indices]
        confidences[tracklet_id] = np.mean(common_label_confidences) if common_label_confidences else 0.0
    
    # Add illegible IDs to the results with label -1
    for illegible_id in illegible_ids:
        if illegible_id not in results:  # Ensure we don't overwrite existing predictions
            results[illegible_id] = "-1"
            confidences[illegible_id] = 1.0  # Assign high confidence for illegible classification
    
    avg_inference_time = np.mean(inference_times)
    
    return results, confidences, avg_inference_time

# Evaluation functions
def basic_metrics(y_true, y_pred):
    """Calculate basic classification metrics."""
    acc = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return acc, precision, recall, f1

def mean_average_precision(y_true, y_pred, n_classes):
    """Calculate Mean Average Precision."""
    mAP = 0.0
    valid_classes = 0
    
    for i in range(n_classes):
        y_true_class = [1 if label == i else 0 for label in y_true]
        y_pred_class = [1 if label == i else 0 for label in y_pred]
        
        if sum(y_true_class) > 0:  # Only compute AP for classes that are present
            try:
                mAP += average_precision_score(y_true_class, y_pred_class)
                valid_classes += 1
            except Exception as e:
                print(f"Error calculating AP for class {i}: {e}")
    
    return mAP / valid_classes if valid_classes > 0 else 0.0

def digit_level_accuracy(predictions, ground_truth):
    """Calculate digit-level accuracy for multi-digit numbers."""
    total_digits = 0
    correct_digits = 0
    
    for pred, true in zip(predictions, ground_truth):
        # Convert prediction and ground truth to strings
        pred_str = str(pred)
        true_str = str(true)
        
        # Special case for illegible (value of -1)
        if true == -1:
            continue
            
        for i in range(min(len(pred_str), len(true_str))):
            total_digits += 1
            if pred_str[i] == true_str[i]:
                correct_digits += 1
    
    return (correct_digits / total_digits) * 100 if total_digits > 0 else 0.0

def evaluate_model(results, confidences, ground_truth_file, avg_inference_time=None):
    """Comprehensive model evaluation."""
    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)
    
    # Prepare data for metrics
    y_pred = []
    y_true = []
    conf_scores = []
    
    for tracklet_id, true_label in ground_truth.items():
        if tracklet_id in results:
            y_pred.append(int(results[tracklet_id]))
            y_true.append(int(true_label))
            conf_scores.append(confidences.get(tracklet_id, 0.0))
    
    # Calculate metrics
    acc, precision, recall, f1 = basic_metrics(y_true, y_pred)
    mAP = mean_average_precision(y_true, y_pred, n_classes=10)  # Assuming 10 classes (digits 0-9)
    digit_acc = digit_level_accuracy(y_pred, y_true)
    avg_confidence = np.mean(conf_scores)
    
    # Create a results dictionary
    eval_results = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mean_average_precision": mAP,
        "digit_level_accuracy": digit_acc,
        "average_confidence": avg_confidence,
    }
    
    if avg_inference_time is not None:
        eval_results["avg_inference_time"] = avg_inference_time
    
    # Calculate confusion matrix for further analysis
    cm = confusion_matrix(y_true, y_pred, labels=range(10))
    eval_results["confusion_matrix"] = cm.tolist()
    
    return eval_results, y_true, y_pred

def statistical_significance_test(model_results):
    """Perform statistical significance tests between model pairs."""
    model_names = list(model_results.keys())
    sig_results = {}
    
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            model1 = model_names[i]
            model2 = model_names[j]
            
            if "y_true" in model_results[model1] and "y_pred" in model_results[model1] and \
               "y_true" in model_results[model2] and "y_pred" in model_results[model2]:
                
                # Calculate per-sample accuracy for paired t-test
                y_true1 = model_results[model1]["y_true"]
                y_pred1 = model_results[model1]["y_pred"]
                
                y_true2 = model_results[model2]["y_true"]
                y_pred2 = model_results[model2]["y_pred"]
                
                # Ensure the same samples are being compared
                if len(y_true1) == len(y_true2) and len(y_pred1) == len(y_pred2):
                    # Create binary arrays showing correct/incorrect predictions
                    correct1 = (np.array(y_true1) == np.array(y_pred1)).astype(np.int32)
                    correct2 = (np.array(y_true2) == np.array(y_pred2)).astype(np.int32)
                    
                    # Perform paired t-test
                    t_stat, p_value = stats.ttest_rel(correct1, correct2)
                    
                    sig_results[f"{model1}_vs_{model2}"] = {
                        "t_statistic": t_stat,
                        "p_value": p_value,
                        "significant_at_0.05": p_value < 0.05
                    }
    
    return sig_results

def evaluate_robustness(model, model_name, image_dir, device, categories=None):
    """Evaluate model robustness across different conditions."""
    if categories is None:
        # Default categories for robustness testing
        categories = {
            "blur": "blur",
            "occlusion": "occl",
            "low_light": "dark",
            "motion": "motion"
        }
    
    robustness_results = {}
    
    for category_name, category_tag in categories.items():
        category_images = [f for f in os.listdir(image_dir) if category_tag in f.lower() and f.endswith(('.jpg', '.png'))]
        
        if not category_images:
            robustness_results[category_name] = {"accuracy": None, "sample_count": 0}
            continue
        
        correct = 0
        confidences = []
        
        for img_name in category_images:
            image_path = os.path.join(image_dir, img_name)
            
            # Extract the true label from filename if available or use a placeholder
            try:
                true_label = int(img_name.split('_')[1])
            except (IndexError, ValueError):
                true_label = None
            
            pred_label, confidence, _ = predict_with_confidence(model, image_path, device)
            confidences.append(confidence)
            
            if true_label is not None and pred_label == true_label:
                correct += 1
        
        if true_label is not None:  # Only calculate accuracy if we have true labels
            accuracy = (correct / len(category_images)) * 100
        else:
            accuracy = None
        
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        robustness_results[category_name] = {
            "accuracy": accuracy,
            "average_confidence": avg_confidence,
            "sample_count": len(category_images)
        }
    
    return robustness_results

def calculate_efficiency_metrics(models):
    """Calculate efficiency metrics for the models."""
    efficiency_results = {}
    
    for name, model in models.items():
        # Count parameters
        num_params = count_parameters(model)
        
        # Calculate model size in MB
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        model_size_mb = param_size / (1024 * 1024)
        
        # Measure inference time on a dummy input (averaged over multiple runs)
        device = next(model.parameters()).device
        dummy_input = torch.randn(1, 3, 256, 256).to(device)
        
        times = []
        for _ in range(10):  # Average over 10 runs
            start_time = time.time()
            with torch.no_grad():
                _ = model(dummy_input)
            times.append(time.time() - start_time)
        
        avg_inference_time = np.mean(times[1:]) * 1000  # Convert to ms, skipping the first run
        
        efficiency_results[name] = {
            "parameters": num_params,
            "model_size_mb": model_size_mb,
            "average_inference_time_ms": avg_inference_time
        }
    
    return efficiency_results

def plot_results(results, output_dir):
    """Generate plots for visualization of results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Accuracy comparison
    plt.figure(figsize=(10, 6))
    model_names = list(results.keys())
    accuracies = [results[model]["accuracy"] for model in model_names]
    
    plt.bar(model_names, accuracies)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Model')
    plt.ylim(0, 100)
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'))
    plt.close()
    
    # Plot 2: Precision, Recall, F1 Score
    plt.figure(figsize=(12, 6))
    metrics = ['precision', 'recall', 'f1_score']
    x = np.arange(len(model_names))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in model_names]
        plt.bar(x + i*width, values, width, label=metric)
    
    plt.title('Precision, Recall, and F1 Score Comparison')
    plt.xticks(x + width, model_names)
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'precision_recall_f1.png'))
    plt.close()
    
    # Plot 3: Inference Time
    plt.figure(figsize=(10, 6))
    times = [results[model].get("avg_inference_time", 0) for model in model_names]
    
    plt.bar(model_names, times)
    plt.title('Inference Time Comparison')
    plt.ylabel('Time (s)')
    plt.xlabel('Model')
    plt.savefig(os.path.join(output_dir, 'inference_time.png'))
    plt.close()
    
    # Plot 4: Digit-Level Accuracy
    plt.figure(figsize=(10, 6))
    digit_accuracies = [results[model].get("digit_level_accuracy", 0) for model in model_names]
    
    plt.bar(model_names, digit_accuracies)
    plt.title('Digit-Level Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Model')
    plt.ylim(0, 100)
    plt.savefig(os.path.join(output_dir, 'digit_level_accuracy.png'))
    plt.close()
    
    # Plot 5: Confidence Scores
    plt.figure(figsize=(10, 6))
    confidences = [results[model].get("average_confidence", 0) for model in model_names]
    
    plt.bar(model_names, confidences)
    plt.title('Average Confidence Score Comparison')
    plt.ylabel('Confidence')
    plt.xlabel('Model')
    plt.ylim(0, 1)
    plt.savefig(os.path.join(output_dir, 'confidence_scores.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Comprehensive evaluation of jersey number recognition models')
    parser.add_argument('--model_paths', nargs='+', required=True, help='Paths to trained models')
    parser.add_argument('--model_names', nargs='+', required=True, help='Names of the models (e.g., baseline, fpn, squeeze)')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory of cropped images')
    parser.add_argument('--illegible_file', type=str, required=True, help='Path to illegible.json file')
    parser.add_argument('--ground_truth_file', type=str, required=True, help='Path to ground truth JSON file')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Directory to save results')
    parser.add_argument('--robustness_dir', type=str, help='Directory with images for robustness testing')
    
    args = parser.parse_args()
    
    # Ensure number of model paths and names match
    if len(args.model_paths) != len(args.model_names):
        raise ValueError("Number of model paths and model names must match!")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize results dictionary
    all_results = {}
    raw_results = {}
    loaded_models = {}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Evaluate each model
    for i, (model_path, model_name) in enumerate(zip(args.model_paths, args.model_names)):
        print(f"\nEvaluating {model_name} model...")
        
        # Load model
        model = load_model(model_path, model_name, device)
        loaded_models[model_name] = model
        
        # Run inference
        print("Running batch inference...")
        results, confidences, avg_inference_time = batch_inference(model, args.image_dir, args.illegible_file, device)
        
        # Evaluate model
        print("Calculating metrics...")
        eval_results, y_true, y_pred = evaluate_model(results, confidences, args.ground_truth_file, avg_inference_time)
        
        # Store results
        all_results[model_name] = eval_results
        raw_results[model_name] = {
            "y_true": y_true,
            "y_pred": y_pred
        }
        
        # Print results
        print(f"Results for {model_name}:")
        print(f"  Accuracy: {eval_results['accuracy']:.2f}%")
        print(f"  Precision: {eval_results['precision']:.4f}")
        print(f"  Recall: {eval_results['recall']:.4f}")
        print(f"  F1-Score: {eval_results['f1_score']:.4f}")
        print(f"  Mean Average Precision (mAP): {eval_results['mean_average_precision']:.4f}")
        print(f"  Digit-Level Accuracy: {eval_results['digit_level_accuracy']:.2f}%")
        print(f"  Average Confidence: {eval_results['average_confidence']:.4f}")
        if "avg_inference_time" in eval_results:
            print(f"  Average Inference Time: {eval_results['avg_inference_time']:.6f} s")
    
    # Calculate efficiency metrics
    print("\nCalculating efficiency metrics...")
    efficiency_results = calculate_efficiency_metrics(loaded_models)
    
    # Run robustness evaluation if directory is provided
    if args.robustness_dir and os.path.exists(args.robustness_dir):
        print("\nEvaluating model robustness...")
        robustness_results = {}
        
        for model_name, model in loaded_models.items():
            print(f"Testing robustness for {model_name}...")
            robustness = evaluate_robustness(model, model_name, args.robustness_dir, device)
            robustness_results[model_name] = robustness
    else:
        robustness_results = None
    
    # Run statistical significance tests
    if len(raw_results) > 1:
        print("\nPerforming statistical significance tests...")
        significance_results = statistical_significance_test(raw_results)
        
        for comparison, result in significance_results.items():
            print(f"  {comparison}:")
            print(f"    t-statistic: {result['t_statistic']:.4f}")
            print(f"    p-value: {result['p_value']:.6f}")
            print(f"    Significant at α=0.05: {result['significant_at_0.05']}")
    else:
        significance_results = None
    
    # Save all results to JSON
    final_results = {
        "model_evaluation": all_results,
        "efficiency_metrics": efficiency_results,
        "statistical_significance": significance_results,
        "robustness_evaluation": robustness_results
    }
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    # Create a JSON-serializable version of the results
    serializable_results = json.loads(json.dumps(final_results, default=convert_to_serializable))
    
    with open(os.path.join(args.output_dir, 'comprehensive_evaluation_results.json'), 'w') as f:
        json.dump(serializable_results, f, indent=4)
    
    print(f"\nResults saved to {os.path.join(args.output_dir, 'comprehensive_evaluation_results.json')}")
    
    # Generate plots
    print("\nGenerating visualization plots...")
    plot_results(all_results, args.output_dir)
    print(f"Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()
