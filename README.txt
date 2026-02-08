Cross-Modal Occlusion Reasoning using Depth-from-Monocular Vision

This project integrates object detection (YOLOv8) and depth estimation (MiDaS) to reason about occlusion in real-world traffic scenes.  
From a single RGB image, the system detects vehicles, estimates their relative depth, and determines which ones are visible (in front) or occluded (behind).  
This approach enhances visual understanding for applications like autonomous driving, traffic surveillance, and intelligent transportation systems.


Key Features
YOLOv8n – detects vehicle classes such as cars, buses, trucks, and motorcycles  
MiDaS_small – estimates depth from a single monocular image  
Cross-modal reasoning – compares median depth values of overlapping objects  
Visualization output – green boxes for visible vehicles, red for occluded  
Summary table – lists each detected object with class, depth, and status  


Tools & Environment
Language: Python 3.10  
Libraries: torch, ultralytics, opencv-python, matplotlib, numpy`  
Hardware: Supports GPU acceleration (CUDA) if available  
Input: RGB traffic scene images (custom or dataset-based)


How to Run
1. Place your test image in the `input/` directory.  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3.Run the main script:
   python main.py
4.Check the outputs/ folder for:
   occlusion_result.jpg (vehicles color-coded by visibility)
   depth_map.jpg (color-coded depth visualization)