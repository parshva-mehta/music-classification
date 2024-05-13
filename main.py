import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.model_selection import GridSearchCV


    
def view_waveform(path):
    data, sr = librosa.load(path)
    
    # Plot specifications
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(data, sr=sr)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    plt.show()



def main():
    # Load the dataframe
    df = pd.read_csv('/Users/parshvamehta/Downloads/Data/features_30_sec.csv')

    df = df.drop(columns=['filename'])
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, Y_train)

    # Predict on the test set
    Y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(Y_test, Y_pred)
    print("Accuracy:", accuracy)

    
    

if __name__ == '__main__':
    main()


