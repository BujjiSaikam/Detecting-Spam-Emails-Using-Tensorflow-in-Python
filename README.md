# Spam Email Detection Using TensorFlow in Python

Author: Bujji Saikam

Project Overview

Spam messages are unsolicited or unwanted emails that flood users' inboxes, often with promotional content. Many email services automatically detect these spam messages, but we can enhance this by building our own machine learning models for spam detection.

This project implements a Spam Email Detection model using TensorFlow. We classify emails as either Spam or Ham (non-spam) by performing Text Classification. The project involves Exploratory Data Analysis (EDA), Text Preprocessing, and developing a Deep Learning model using LSTM layers.


---

Table of Contents

1. Installation


2. Project Structure


3. Dataset


4. Exploratory Data Analysis


5. Text Preprocessing


6. Model Development


7. Model Training


8. Evaluation


9. Results


10. Conclusion




---

Installation

To run this project locally, follow these steps:

1. Clone the repository:

git clone https://github.com/BujjiSaikam/Spam-Email-Detection.git
cd Spam-Email-Detection


2. Install the required dependencies:

pip install -r requirements.txt


3. Run the code:

python spam_email_detection.py




---

Project Structure

Spam-Email-Detection/
│
├── data/
│   └── emails.csv            # Dataset file
│
├── spam_email_detection.py    # Main Python file for model development
├── README.md                  # Project documentation (this file)
├── requirements.txt           # Python libraries required to run the project


---

Dataset

The dataset used for this project contains email texts labeled as either Spam (1) or Ham (0). The dataset has 5728 records, with two columns:

text: The email content

spam: The label (1 for Spam, 0 for Ham)



---

Exploratory Data Analysis

We perform EDA to get insights into the dataset, including:

The distribution of Spam and Ham emails.

Word frequency analysis using WordCloud.


Some code snippets:

sns.countplot(x='spam', data=data)
plt.show()

plot_word_cloud(balanced_data[balanced_data['spam'] == 0], typ='Non-Spam')
plot_word_cloud(balanced_data[balanced_data['spam'] == 1], typ='Spam')


---

Text Preprocessing

Before feeding text data into the model, we perform the following preprocessing steps:

Stopwords removal

Punctuations removal

Word tokenization and padding


balanced_data['text'] = balanced_data['text'].apply(lambda text: remove_stopwords(text))
balanced_data['text'] = balanced_data['text'].apply(lambda x: remove_punctuations(x))


---

Model Development

The model is built using TensorFlow's Sequential API with the following layers:

Embedding Layer: To convert words into dense vectors.

LSTM Layer: To capture sequential patterns in the text data.

Dense Layer: To classify the emails as Spam or Ham.


Model Summary:

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=32, input_length=max_len))
model.add(tf.keras.layers.LSTM(16))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.summary()


---

Model Training

The model is trained using Binary Crossentropy as the loss function and Adam optimizer. We use EarlyStopping and ReduceLROnPlateau callbacks to monitor the model's performance.

es = EarlyStopping(patience=3, monitor='val_accuracy', restore_best_weights=True)
lr = ReduceLROnPlateau(patience=2, monitor='val_loss', factor=0.5, verbose=0)

history = model.fit(train_sequences, train_Y, validation_data=(test_sequences, test_Y), epochs=20, batch_size=32, callbacks=[lr, es])


---

Evaluation

The model achieves an accuracy of 97.62% on the test data. Below is the performance graph showing the training and validation accuracy over epochs:

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()


---

Results

Training Accuracy: 97.44%

Test Accuracy: 97.62%

Test Loss: 0.109



---

Conclusion

The spam detection model developed in this project successfully classifies emails with a high degree of accuracy using LSTM layers. Further improvements can be made by experimenting with other text representations (e.g., BERT) and optimizing hyperparameters.


---

Thank You!
