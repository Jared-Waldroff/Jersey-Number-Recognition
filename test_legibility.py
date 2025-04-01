import os
import torch
import json
import numpy as np
import argparse
from pathlib import Path
import configuration as cfg
import legibility_classifier as lc
from networks import MyLegibilityConvNet, LegibilityClassifier34
from flexible_networks import FlexibleLegibilityConvNet, AdaptiveConvNet

def main():
    parser = argparse.ArgumentParser(description='Test Jersey Number Legibility Classifier')
    parser.add_argument('--input_dir', type=str, 
                        default='/Users/aadilshaji/Desktop/University/Studies/4th year/Jersey-Number-Recognition/data/SoccerNet/jersey-2023/extracted/test/images',
                        help='Directory containing tracklet folders with images')
    parser.add_argument('--model_path', type=str, 
                        default=os.path.join('/Users/aadilshaji/Desktop/University/Studies/4th year/Jersey-Number-Recognition/data/SoccerNet/jersey-2023', 
                                          cfg.dataset['SoccerNet']['legibility_model']),
                        help='Path to the pre-trained legibility model')
    parser.add_argument('--output_dir', type=str, 
                        default='/Users/aadilshaji/Desktop/University/Studies/4th year/Jersey-Number-Recognition/data/SoccerNet/jersey-2023/out/legibility_results',
                        help='Output directory for results')
    parser.add_argument('--arch', type=str, default='resnet34',
                        choices=['resnet34', 'myconvnet', 'flexible', 'adaptive'],
                        help='Model architecture to use')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    parser.add_argument('--tracklet', type=str, default=None,
                        help='Test a specific tracklet (folder name)')
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Using device: {device}")

    # Determine which tracklets to process
    if args.tracklet:
        # Process a single tracklet
        tracklets = [args.tracklet]
    else:
        # Process all tracklets in the input directory
        tracklets = [t for t in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, t))]
    
    print(f"Found {len(tracklets)} tracklets to process")
    
    # Results dictionary
    results = {}
    
    # Process each tracklet
    for tracklet in tracklets:
        tracklet_path = os.path.join(args.input_dir, tracklet)
        
        if not os.path.isdir(tracklet_path):
            continue
            
        # Get all image files in the tracklet directory
        image_files = [os.path.join(tracklet_path, img) for img in os.listdir(tracklet_path) 
                      if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"No images found in tracklet {tracklet}")
            continue
            
        print(f"Processing tracklet {tracklet} with {len(image_files)} images")
        
        # Run the legibility classifier
        try:
            track_results = lc.run(image_files, args.model_path, 
                                 arch=args.arch, threshold=args.threshold)
            
            # Get indices of legible images
            legible_indices = list(np.nonzero(track_results)[0])
            legible_image_names = [os.path.basename(image_files[i]) for i in legible_indices]
            
            # Store results
            results[tracklet] = {
                'legible_count': len(legible_indices),
                'total_images': len(image_files),
                'legible_percentage': len(legible_indices) / len(image_files) * 100 if image_files else 0,
                'legible_images': legible_image_names
            }
            
            print(f"Tracklet {tracklet}: {len(legible_indices)}/{len(image_files)} images are legible ({results[tracklet]['legible_percentage']:.2f}%)")
            
        except Exception as e:
            print(f"Error processing tracklet {tracklet}: {e}")
    
    # Save results to file
    results_file = os.path.join(args.output_dir, f"legibility_results_{args.arch}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {results_file}")
    
    # Summary
    legible_tracklets = [t for t, r in results.items() if r['legible_count'] > 0]
    print(f"\nSummary:")
    print(f"Total tracklets processed: {len(results)}")
    print(f"Tracklets with legible images: {len(legible_tracklets)} ({len(legible_tracklets) / len(results) * 100 if results else 0:.2f}%)")

if __name__ == "__main__":
    main()
