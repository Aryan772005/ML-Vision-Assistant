# ML-Vision-Assistant

ML-Vision-Assistant is an advanced computer vision application that seamlessly merges real-time Sign Language Detection with Face Shape and Emotion analysis. 

Built for performance and a modern UI experience, it features an intelligent conversational AI that translates your hand signs into spoken words fluently, while simultaneously tracking your facial proportions to offer dynamic style recommendations.

## Features

- **Real-Time Sign Language Translation:** Instantly converts hand gestures into text using a custom-trained Random Forest model and MediaPipe hand tracking.
- **Fluent Auto-Speech:** Automatically speaks completed sentences aloud using integrated Text-to-Speech (TTS).
- **Face Shape & Emotion Detection:** Analyzes your facial landmarks continuously in the background to determine your face shape (Oval, Square, Heart, Oblong) and current emotion.
- **Integrated AI Assistant:** A built-in AI assistant to chat with, receive style recommendations based on your face shape, and summarize your signed conversations.
- **Unified Dashboard:** A sleek, dark-themed UI built with CustomTkinter.

---

##  Core ML Architecture

This application is built on a robust, multi-tiered Artificial Intelligence and Machine Learning architecture:

1. **Classical Machine Learning (Scikit-Learn):** Uses a highly optimized `RandomForestClassifier` trained on custom landmark datasets to classify complex sign language vocabulary with incredible accuracy and low latency.
2. **Deep Learning Neural Networks (MediaPipe):** Integrates Google's advanced CNNs for real-time 3D hand tracking and facial landmark extraction, ensuring flawless performance even on standard webcams.
3. **Generative AI (Large Language Models):** Powered by an integrated Gemini NLP backend, the system seamlessly fixes broken grammatical structures from raw signs, processes facial style recommendations, and engages in contextual chat.

---

##  How to Install and Run( Install and give a reviu and dont forget to train the model you can use sme from huging fa)

It is very easy to set this up on your own device.

### 1. Prerequisites
Ensure you have Python 3.9 or higher installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/yourusername/ML-Vision-Assistant.git
cd ML-Vision-Assistant
```

### 3. Install Dependencies
Install all the required Python libraries using pip:
```bash
pip install -r requirements.txt
```

### 4. Set up your API Key
If you want the AI chat and style recommendations to work, create a `.env` file in the root directory and add your Google Gemini API key:
```env
GEMINI_API_KEY=your_api_key_here
```

### 5. Run the Application
```bash
python main.py
```

---

##  How to Train Your Own Model

ML-Vision-Assistant allows you to completely customize the sign language vocabulary by capturing your own data and training a new model.

### Step 1: Collect Data
Run the data collection script. This will turn on your webcam and prompt you to perform the hand signs you want the model to learn.
```bash
python dataset/collect_data.py
```
*Follow the on-screen terminal instructions to capture datasets for different classes (e.g., A, B, C, Hello, Thanks, Space).*

### Step 2: Train the Model
Once you have collected enough data, run the training script. This script will extract the hand landmarks from your captured images and train a Random Forest Classifier.
```bash
python dataset/train_model.py
```
*When the training is complete, a new `model.p` file will be generated.*

### Step 3: Run the App
Start the main application again. It will automatically load your newly trained `model.p`!
```bash
python main.py
```
