# load and evaluate a saved model
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import cv2

from skimage.metrics import structural_similarity as compare_ssim

model = load_model('./metal_nut_save/metal_nut_binary100.h5')

img_path = './dataset/metal_nut_test/010.jpg'
img_size_weight = 512
img_size_height = 512

# img = Image.open(img_path)
# cv2.INTER_AREA 영상 축소 시
# cv2.INTER_LINEAR < cv2.INTER_CUBIC < cv2.INTER_LANCZOS4
img = cv2.imread(img_path, 0)
img = cv2.resize(img, dsize=(img_size_weight, img_size_height), interpolation=cv2.INTER_LANCZOS4)
img = np.array(img, dtype=np.float32) / 255.0
img = np.reshape(img, (1, img_size_weight, img_size_height, 1))

ax = plt.subplot(2, 1, 1)
plt.imshow(img.reshape((img_size_weight, img_size_height, 1)))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

prediction = model.predict(img)
ax = plt.subplot(2, 1, 2)
plt.imshow(prediction.reshape(img_size_weight, img_size_height, 1))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.show()

img = np.squeeze(img)
prediction = np.squeeze(prediction)
score, diff = compare_ssim(img, prediction, full=True)
# full=True: 이미지 전체에 대해서 구조비교를 수행한다.
diff = (diff * 255).astype('uint8')
print(f'SSIM: {score:.6f}')
plt.imshow(diff)
plt.show()
# option parameter use cv2.THRESH_BINARY_INV
retval, result = cv2.threshold(diff, 128, 255, cv2.THRESH_BINARY_INV)
plt.imshow(result)
plt.show()


