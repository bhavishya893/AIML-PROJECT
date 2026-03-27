# AIML-PROJECT

Deepfake Video Detection

Overview
This project detects whether a video is real or deepfake by analyzing frame-to-frame consistency. It uses image similarity metrics (MSE & SSIM) to identify unnatural smoothness or inconsistencies commonly found in deepfake videos.

Features
-Extracts frames from video using OpenCV
-Skips frames for faster processing
-Compares consecutive frames using: MSE (Mean Squared Error) & SSIM (Structural Similarity Index)
-Detects deepfake based on similarity thresholds
-Generates visual analysis plots
-Outputs confidence score

How it works
-Extracts frames from the video
-Compares MSE & SSIM(Deepfakes often have: low MSE & high SSIM)
-Predicts whether the video is deepfake or not

How to Run
-Open terminal or command prompt.
-Navigate to the folder where main.py is saved.
-Run the program using:python

Name: Bhavishya     
Registration No: 25BCE10893
