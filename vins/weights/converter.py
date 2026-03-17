#!/usr/bin/env python3
"""
YOLO11 Model Converter
Converts YOLO11 .pt weights to both ONNX and TensorRT formats
for dual CPU/GPU deployment strategy.
"""

import argparse
import sys
from pathlib import Path
import torch

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics package not found.")
    print("Install with: pip install ultralytics")
    sys.exit(1)


def convert_to_onnx(model_path: str, output_dir: str = "weights", imgsz: int = 640, simplify: bool = True):
    """
    Convert YOLO11 .pt model to ONNX format for CPU inference.
    
    Args:
        model_path: Path to the .pt weights file
        output_dir: Directory to save the exported model
        imgsz: Input image size (default: 640)
        simplify: Whether to simplify the ONNX model (default: True)
    """
    print(f"\n{'='*60}")
    print("Converting to ONNX format (for CPU/ONNX Runtime)")
    print(f"{'='*60}")
    
    try:
        model = YOLO(model_path)
        
        # Export to ONNX
        export_path = model.export(
            format='onnx',
            imgsz=imgsz,
            simplify=simplify,
            dynamic=False,  # Static shape for better optimization
            opset=12  # ONNX opset version
        )
        
        print(f"? ONNX model exported successfully: {export_path}")
        return export_path
        
    except Exception as e:
        print(f"? Error during ONNX export: {e}")
        return None


def convert_to_tensorrt(model_path: str, output_dir: str = "weights", imgsz: int = 640, half: bool = True):
    """
    Convert YOLO11 .pt model to TensorRT format for GPU inference.
    
    Args:
        model_path: Path to the .pt weights file
        output_dir: Directory to save the exported model
        imgsz: Input image size (default: 640)
        half: Use FP16 precision (default: True for better performance)
    """
    print(f"\n{'='*60}")
    print("Converting to TensorRT format (for NVIDIA GPU)")
    print(f"{'='*60}")
    
    try:
        model = YOLO(model_path)
        
        # Export to TensorRT
        device = "0" if torch.cuda.is_available() else "cpu"

        export_path = model.export(
            format='engine',
            imgsz=imgsz,
            half=half,
            device=device,
            workspace=4,
            simplify=True
        )

        
        
        # export_path = model.export(
        #     format='engine',
        #     imgsz=imgsz,
        #     half=half,  # FP16 for better performance
        #     device=0,  # GPU device
        #     workspace=4,  # Max workspace size in GB
        #     simplify=True
        # )
        
        print(f"? TensorRT model exported successfully: {export_path}")
        return export_path
        
    except Exception as e:
        print(f"? Error during TensorRT export: {e}")
        print("Note: TensorRT export requires NVIDIA GPU and TensorRT installation")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO11 models to ONNX and TensorRT formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert for both CPU and GPU deployment
  python convert_yolo11.py yolo11s-seg.pt
  
  # Specify custom image size
  python convert_yolo11.py yolo11s-seg.pt --imgsz 1280
  
  # Convert only to ONNX (skip TensorRT)
  python convert_yolo11.py yolo11s-seg.pt --onnx-only
  
  # Convert only to TensorRT (skip ONNX)
  python convert_yolo11.py yolo11s-seg.pt --tensorrt-only
        """
    )
    
    parser.add_argument(
        'model_path',
        type=str,
        help='Path to YOLO11 .pt weights file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='weights',
        help='Output directory for converted models (default: weights)'
    )
    
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size (default: 640)'
    )
    
    parser.add_argument(
        '--onnx-only',
        action='store_true',
        help='Convert only to ONNX format'
    )
    
    parser.add_argument(
        '--tensorrt-only',
        action='store_true',
        help='Convert only to TensorRT format'
    )
    
    parser.add_argument(
        '--no-half',
        action='store_true',
        help='Disable FP16 precision for TensorRT (use FP32)'
    )
    
    parser.add_argument(
        '--no-simplify',
        action='store_true',
        help='Disable ONNX model simplification'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.model_path).exists():
        print(f"Error: Model file '{args.model_path}' not found")
        sys.exit(1)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\nModel: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Image size: {args.imgsz}")
    
    results = {}
    
    # Convert to ONNX (unless tensorrt-only flag is set)
    if not args.tensorrt_only:
        onnx_path = convert_to_onnx(
            args.model_path,
            args.output_dir,
            args.imgsz,
            simplify=not args.no_simplify
        )
        results['onnx'] = onnx_path
    
    # Convert to TensorRT (unless onnx-only flag is set)
    if not args.onnx_only:
        tensorrt_path = convert_to_tensorrt(
            args.model_path,
            args.output_dir,
            args.imgsz,
            half=not args.no_half
        )
        results['tensorrt'] = tensorrt_path
    
    # Summary
    print(f"\n{'='*60}")
    print("Conversion Summary")
    print(f"{'='*60}")
    
    for format_name, path in results.items():
        status = "? Success" if path else "? Failed"
        print(f"{format_name.upper():12} : {status}")
        if path:
            print(f"             {path}")
    
    print(f"\n{'='*60}")
    print("Next Steps:")
    print(f"{'='*60}")
    print("1. Copy the generated model files to your C++ project")
    print("2. Use TensorRT engine for GPU inference (CUDA)")
    print("3. Use ONNX model for CPU inference (ONNX Runtime)")
    print("4. Implement runtime detection to choose the appropriate model")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()