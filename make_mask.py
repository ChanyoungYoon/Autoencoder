import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import matplotlib.pyplot as plt

# img1 : 원본 영상, img2 : 비교 영상
start_num = 601
max_num = 617
img1_path = './dataset/T6/'
img2_path = './dataset/tomato/test/print/'
save_path = './dataset/tomato/ground_truth/print/'

# 이미지 이름 4자리의 숫자일 경우만 읽을 수 있음
for i in range(start_num, max_num + 1):

    # 영상 gray로 변환
    img1 = cv2.imread(img1_path + str(i).zfill(4) + '.png')
    img2 = cv2.imread(img2_path + str(i).zfill(4) + '.png')
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img = np.squeeze(gray1)
    img1 = np.squeeze(gray2)
    score, diff = compare_ssim(gray1, gray2, full=True)
    # full=True: 이미지 전체에 대해서 구조비교를 수행한다.
    diff = (diff * 255).astype('uint8')
    print(f'SSIM: {score:.6f}')

    # option parameter cv2.THRESH_BINARY+cv2.THRESH_OTSU or cv2.THRESH_BINARY_INV
    _, result = cv2.threshold(diff, 128, 255, cv2.THRESH_BINARY_INV)
    result = cv2.resize(result, (1024,1024))
    my_dpi = 96
    plt.figure(figsize=(1024/my_dpi, 1024/my_dpi), dpi=my_dpi)
    ax = plt.subplot(1, 1, 1)
    # cmap변경 (default는 보라/노랑)
    plt.imshow(result, cmap='gray')
    # x축 y축 값 지우기
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.savefig(save_path + str(i).zfill(4) + '_mask' + '.png', dpi=my_dpi)
