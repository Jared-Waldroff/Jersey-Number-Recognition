import re
import ast
import matplotlib.pyplot as plt

# Path to the training log file
log_file_path = r"C:\Users\jared\PycharmProjects\CLIP4STR\output\vl4str_2025-04-04_11-16-31\train.log"

# Initialize dictionaries to store metrics by epoch
train_metrics = {}
val_metrics = {}

# Regular expressions to match training and validation metric lines
train_pattern = re.compile(r"Epoch (\d+) Training Metrics: (.+)")
val_pattern = re.compile(r"Epoch (\d+) Validation Metrics: (.+)")

# Read and parse the log file
with open(log_file_path, "r") as f:
    for line in f:
        line = line.strip()
        # Match training metrics
        train_match = train_pattern.search(line)
        if train_match:
            epoch = int(train_match.group(1))
            # Convert string dictionary to a Python dict safely
            metrics_dict = ast.literal_eval(train_match.group(2))
            train_metrics[epoch] = metrics_dict
        # Match validation metrics
        val_match = val_pattern.search(line)
        if val_match:
            epoch = int(val_match.group(1))
            metrics_dict = ast.literal_eval(val_match.group(2))
            val_metrics[epoch] = metrics_dict

# Sort epochs
epochs = sorted(set(list(train_metrics.keys()) + list(val_metrics.keys())))

# Extract values for each metric. Some epochs might only have one type.
train_loss = [train_metrics.get(ep, {}).get("loss", None) for ep in epochs]
val_loss   = [val_metrics.get(ep, {}).get("val_loss", None) for ep in epochs]
val_accuracy = [val_metrics.get(ep, {}).get("val_accuracy", None) for ep in epochs]
hp_metric    = [val_metrics.get(ep, {}).get("hp_metric", None) for ep in epochs]

# Plot Training & Validation Loss
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, marker='o', label='Training Loss')
plt.plot(epochs, val_loss, marker='o', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss vs Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Validation Accuracy & HP Metric
plt.figure(figsize=(8, 5))
plt.plot(epochs, val_accuracy, marker='o', label='Validation Accuracy')
plt.plot(epochs, hp_metric, marker='o', label='HP Metric')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.title('Validation Accuracy & HP Metric vs Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
