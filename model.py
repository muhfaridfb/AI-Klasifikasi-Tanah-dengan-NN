import tensorflow as tf  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense, Dropout  
from tensorflow.keras.utils import to_categorical  

import pandas as pd             
import numpy as np              
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import LabelEncoder, MinMaxScaler  


from google.colab import drive
drive.mount('/content/drive')

import pandas as pd


file_path = '/content/drive/My Drive/semester 5/AI/tugas akhir/soil.csv'


df = pd.read_csv(file_path)
print(df.head())

df = df.drop_duplicates()

print(df.describe())


df_selected = df[['N','P','K', 'humidity', 'ph', 'label']]
print(df_selected.head())




df_selected = df[['N','P','K', 'humidity', 'ph', 'label']].copy()


print("Unique labels:", df_selected['label'].unique())


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_selected['label'] = le.fit_transform(df_selected['label'])

print(df_selected.head())



df_sampled = df_selected.sample(frac=0.9, random_state=42)


X = df_sampled.drop('label', axis=1)
y = df_sampled['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cek jumlah data
print("Data latih:", X_train.shape)
print("Data uji:", X_test.shape)


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



import pandas as pd


X_train_np = X_train  
y_train_np = y_train

train_data = pd.DataFrame(X_train_np, columns=[f"feature_{i+1}" for i in range(X_train_np.shape[1])])
train_data['label'] = y_train_np

print(train_data.head())



# Normalisasi fitur
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Define the neural network model
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),  
    Dropout(0.5),  
    Dense(64, activation='relu'),                              
    Dropout(0.3),  
    Dense(len(y.unique()), activation='softmax')                
])

# Compile the model
model.compile(
    optimizer='adam',              
    loss='sparse_categorical_crossentropy',  
    metrics=['accuracy']              
)
from tensorflow.keras.callbacks import EarlyStopping

early_stopping_monitor = EarlyStopping(
    monitor='val_loss',
    patience=10,  
    restore_best_weights=True  
    
)


history = model.fit(
    X_train, y_train,
    validation_split=0.2,  
    epochs=500,             
    batch_size=32          
)


test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc*100:.2f}%")
