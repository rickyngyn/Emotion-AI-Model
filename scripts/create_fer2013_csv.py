import os
import cv2
import pandas as pd

train_dir = '../data/train'
test_dir = '../data/test'

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def processImage(data_dir, usage):
    data = []
    for emotion in emotions:
        folder_path = os.path.join(data_dir, emotion)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            try: 
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (48,48))
                pixels = img.flatten()
                pixel_str = ' '.join(str(p) for p in pixels)
                label = emotions.index(emotion)
                data.append([pixel_str, label, usage])
            except:
                print(f"Skipping: {img_path}")
    return data

train_data = processImage(train_dir, 'Training')
test_data = processImage(test_dir, 'Testing')

columns = ['pixels', 'emotion', 'Usage']
df = pd.DataFrame(train_data + test_data, columns=columns)
df.to_csv('../data/fer2013.csv', index=False)
print("fer2013.csv created!")
