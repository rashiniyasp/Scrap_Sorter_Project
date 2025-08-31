# Real-Time Scrap Sorter Simulation

This project is a simulation of an industrial scrap sorting system using computer vision, fulfilling the requirements for the Computer Vision Engineer Intern assignment. It uses a custom-trained YOLOv8 model to detect, classify, and generate pick-points for different types of scrap from a video stream.


## My Approach

I started by selecting a dataset but quickly found it was producing poor results due to data quality issues. My key decision was to create a smaller, high-quality, custom dataset of 50 images each for three classes: cardboard, metal, and plastic. 

Because the dataset was small, I focused on using **heavy data augmentation** during the training process. This involved random rotations, scaling, and color shifts to create a more robust model. The model was trained on Kaggle to leverage their free GPU resources.

The final pipeline is a Python script using OpenCV that reads a video file, runs inference with the custom-trained YOLOv8 model, and overlays the bounding boxes, pick-points, and a live object counter on the video stream.

## Libraries Used
*   PyTorch
*   Ultralytics (for YOLOv8)
*   OpenCV-Python
*   NumPy

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/rashiniyasp/Scrap_Sorter_Project.git
    cd Scrap_Sorter_Project

    ```
2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
3.  **Download the model weights:**
    *Place the `best.pt` file in the main project directory.*

4.  **Run the live inference script:**
    *Make sure your test video (e.g., `test5.mp4`) is in the project folder and the `VIDEO_SOURCE` variable in `live_inference.py` is updated.*
    ```bash
    python live_inference.py
    ```

## Challenges & Learnings

- **Initial Model Failure:** The first model trained on a large public dataset performed poorly, misclassifying objects due to a "domain gap" between the training data and the test video.
- **Data-Centric Approach:** I learned that a smaller, cleaner, more relevant dataset is far more effective. The process of creating a custom dataset gave me full control over quality.
- **Label Correction:** I encountered an `IndexError` due to incorrect class IDs in my label files. I solved this by writing a Python script to automatically correct the labels based on the filenames, which was far more efficient than manual editing.
