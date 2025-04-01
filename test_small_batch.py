import os
import torch
import argparse
import legibility_classifier as lc
from networks import MyLegibilityConvNet, LegibilityClassifier34
from flexible_networks import FlexibleLegibilityConvNet, AdaptiveConvNet, ImprovedAdaptiveConvNet
import configuration as cfg

def main():
    parser = argparse.ArgumentParser(description='Test Jersey Number Legibility Classifier on a small batch')
    parser.add_argument('--input_dir', type=str, 
                        default='/Users/aadilshaji/Desktop/University/Studies/4th year/Jersey-Number-Recognition/data/SoccerNet/jersey-2023/test/images',
                        help='Directory containing tracklet folders with images')
    parser.add_argument('--model_path', type=str, 
                        default=os.path.join('/Users/aadilshaji/Desktop/University/Studies/4th year/Jersey-Number-Recognition/data/SoccerNet/jersey-2023/', 
                                          cfg.dataset['SoccerNet']['legibility_model']),
                        help='Path to the pre-trained legibility model')
    parser.add_argument('--arch', type=str, default='resnet34',
                        choices=['resnet34', 'myconvnet', 'flexible', 'adaptive', 'improved'],
                        help='Model architecture to use')
    parser.add_argument('--num_images', type=int, default=5,
                        help='Number of images to test per tracklet')
    parser.add_argument('--num_tracklets', type=int, default=2,
                        help='Number of tracklets to test')
    args = parser.parse_args()
    
    #print(f"Using device: {torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')}")
    print(f"Using device: {torch.device('mps' if torch.backends.mps.is_available() else 'cuda:0' if torch.cuda.is_available() else 'cpu')}")

    
    # Get a list of tracklet folders
    tracklets = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
    tracklets = tracklets[:args.num_tracklets]  # Limit to specified number
    
    print(f"Testing {args.num_tracklets} tracklets with {args.num_images} images each")
    
    # Process each tracklet
    for tracklet in tracklets:
        tracklet_path = os.path.join(args.input_dir, tracklet)
        
        # Get image files
        image_files = [os.path.join(tracklet_path, img) for img in os.listdir(tracklet_path) 
                      if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files = image_files[:args.num_images]  # Limit to specified number
        
        if not image_files:
            print(f"No images found in tracklet {tracklet}")
            continue
            
        print(f"\nProcessing tracklet {tracklet} with {len(image_files)} images")
        
        # Load the appropriate model
        if args.arch == 'myconvnet':
            model = MyLegibilityConvNet()
        elif args.arch == 'flexible':
            model = FlexibleLegibilityConvNet()
        elif args.arch == 'adaptive':
            model = AdaptiveConvNet()
        elif args.arch == 'improved':
            model = ImprovedAdaptiveConvNet()
        else:  # resnet34
            model = LegibilityClassifier34()
        
        # Print model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel Architecture: {args.arch}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        try:
            # Run the model on the images
            results = lc.run(image_files, args.model_path, arch=args.arch, threshold=0.5, batch_size=2)
            
            # Print results
            legible_count = sum(results)
            print(f"Results for tracklet {tracklet}:")
            print(f"  {legible_count}/{len(image_files)} images are legible ({legible_count/len(image_files)*100:.2f}%)")
            
            # Print individual image results
            for i, (img, res) in enumerate(zip(image_files, results)):
                status = "Legible" if res else "Not legible"
                print(f"  Image {i+1}: {os.path.basename(img)} - {status}")
                
        except Exception as e:
            print(f"Error processing tracklet {tracklet}: {str(e)}")
            print("This error typically occurs when the model architecture doesn't match the saved weights.")
            print("Try using a different architecture with --arch or check if the model path is correct.")

if __name__ == "__main__":
    main()
