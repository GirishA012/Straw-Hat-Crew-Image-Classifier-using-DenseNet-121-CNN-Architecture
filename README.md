# Straw Hat Image Classifier Using DenseNet-121 CNN Architecture  

## 📌 Project Overview  
This project is an image classifier that identifies whether an anime character belongs to the **Straw Hat Pirate Crew** from *One Piece*. The model is built using the **DenseNet-121 CNN architecture** and was trained on a dataset created by collecting images from Google.  

## 📂 Dataset and Preprocessing  
- Images of **10 Straw Hat crew members** were gathered from Google.  
- **Data augmentation** was applied to increase dataset size (available in the repository).  
- The dataset was split into **train and test sets** for model training.  

## 🧠 Model Architecture  
- **DenseNet-121 CNN Architecture** was used for training.  
- Due to **computational limitations**, the model was trained with fewer epochs.  
- The model predicts whether an input image belongs to a **Straw Hat crew member** or not.  
- If the character is a crew member, the model returns their **name**; otherwise, it displays **"Not a member of the Straw Hat crew"**.  

## 📊 Performance and Challenges  
- The **accuracy is lower** due to a **small dataset** and **limited computational power**.  
- A larger dataset is required for better accuracy.  

## 🌐 Deployment  
- This project was hosted on **Streamlit** for interactive image classification.  
- Users could upload an image, and the model would classify the character.  
- The deployment code is available in the repository.  

## 🚀 Technologies Used  
- **Python**  
- **Libraries:**  
  - Pandas, NumPy (Data Handling)  
  - OpenCV, Pillow (Image Processing)  
  - TensorFlow/Keras (Deep Learning)  
  - DenseNet-121 (CNN Architecture)  
  - Streamlit (Web App Deployment)  

## 🏁 Conclusion  
This project demonstrates the application of **CNNs for anime character classification**, particularly focusing on *One Piece*. While the accuracy is limited due to dataset size and computational power, the approach showcases how deep learning can be used for character recognition.  

---

