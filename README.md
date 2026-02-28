!!Important Things
background must be black only!!!
    
    🌾 Rice Quality Inspection AI (Web App)
A real-time computer vision solution designed to automate the quality control and grading of rice grains. This application utilizes OpenCV for advanced image processing and Streamlit for an interactive, web-based user interface. The system classifies rice grains into four distinct categories: Good, Broken, Spoiled (Yellowed), and Foreign Objects.

    ✨ Key Features
Real-time Detection: Seamlessly connects to a webcam for live quality analysis.
Image Upload Support: Allows users to upload static images (JPG, PNG) for batch processing or detailed inspection.
Dynamic Parameter Tuning: Features a sidebar to adjust Grain Separation (Watershed sensitivity) and Spoiled Sensitivity (HSV thresholding) in real-time.
Live Quality Dashboard: Displays grain counts, classification breakdown, and an overall quality percentage.
Exportable Results: One-click download button to save processed images with bounding boxes and labels for reporting.

    🛠 Tech Stack
Python: Primary programming language.
OpenCV: Core library for Image Processing and the Watershed segmentation algorithm.
Streamlit: Framework for building the web application and interactive dashboard.
NumPy: Used for high-performance numerical computation and image array manipulation.

    🚀 Installation & Usage
    1. Clone the repository
git clone https://github.com/YourUsername/Rice-Quality-Inspection-AI.git
cd Rice-Quality-Inspection-AI
            
    2. Install Dependencies
pip install -r requirements.txt

    3. Run the App
streamlit run app.py

    👥 Developers
TanapatSamudinkaew
kenkunanon
Nawattakorn
Chris Pheraphon
