# імпортуємо потрібні модулі
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# побудуємо парсер аргументів, передаймо аргументи
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to output face mask detector model")
args = vars(ap.parse_args())

# ініціалізужмо початковий рівень тренування, кількість епох тренування,
# та розмір партії картинок
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# беремо список картинок з відповідної директорії , потім ініціалізуємо
# список даних та класи зображень
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []
# пропустимо через цикл всі шляхи до картинок
for imagePath in imagePaths:
	# добудемо лейбл класу з назви файлу
	label = imagePath.split(os.path.sep)[-2]
	# загрузимо вхідну картинку (224х224) та опрацюємо її
	image = load_img(imagePath, target_size=(224, 224))
	image = img_to_array(image)
	image = preprocess_input(image)
	# обновимо списки даних та лейблів
	data.append(image)
	labels.append(label)
# конвертуємо списки даних та лейблів в масив NumPy
data = np.array(data, dtype="float32")
labels = np.array(labels)

# виконаємо одноразове кодування на лейблах
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
# розділемо дані у відношенні 80 : 20 для тренування
# та тестування відповідно
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)
# побудуємо аугментатор зображення для тренування мережі
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# загрузимо мережу MobileNetV2, впевнимося що ми не використовуємо 
# голову моделі
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))
# побудуємо нашу голову моделі, яка буде встановленa на MobileNetV2
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)
# вставновлюємо голову на базову модель (це стане
# моделлю яку ми і будемо тренувати)
model = Model(inputs=baseModel.input, outputs=headModel)
# пропустимо через цикл всі слої базової моделі, і заморозимо їх,
# щоб вони не оновлювались під час тренування
for layer in baseModel.layers:
	layer.trainable = False

# компілюємо модель
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])
# тренуємо голову мережі
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# робимо передбачення щодо тренувального сету зображень
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)
# для кожного зображення в тестовому сеті нам потрібно отримамати індекс 
# лейбла з найбільшою відповідною передбаченною відповідностю
predIdxs = np.argmax(predIdxs, axis=1)
# показуємо відформатовані результати тренування моделі
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))
# зберігаємо модель
print("[INFO] saving mask detector model...")
model.save(args["model"], save_format="h5")

# побудуємо графік втрат та точності
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

