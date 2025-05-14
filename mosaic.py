import cv2
import numpy as np
import os
from tqdm import tqdm


def get_image_mean(image):
    """计算图像的均值"""
    return np.mean(image)


def sort_small_images(small_images):
    """根据图像的均值排序小图"""
    small_images_with_means = [(get_image_mean(img), img) for img in small_images]
    small_images_with_means.sort(key=lambda x: x[0])  # 按均值排序
    return list(zip(*small_images_with_means))


def find_closest_index(arr, target):
    n = len(arr)
    if target <= arr[0]:
        return 0
    if target >= arr[-1]:
        return n - 1

    # 二分查找
    low, high = 0, n - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1

    # 此时 low 和 high 是目标值的最近边界
    # 比较 arr[low] 和 arr[high] 哪个更接近 target
    if low < n and high >= 0:
        if abs(arr[low] - target) < abs(arr[high] - target):
            return low
        else:
            return high
    elif low < n:
        return low
    else:
        return high


def find_best_match(target_block, sorted_small_images):
    """在排序后的小图集合中使用折半查找找到与目标块最匹配的小图"""
    target_mean = get_image_mean(target_block)

    closest_index = find_closest_index(sorted_small_images[0], target_mean)

    closest_image = sorted_small_images[1][closest_index]
    return closest_image


def get_image_blocks(image, block_size):
    """将图像分割成多个小块"""
    blocks = []
    height, width, _ = image.shape
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = image[y:y + block_size, x:x + block_size]
            blocks.append((x, y, block))
    return blocks


def process_block(block_data, sorted_small_images, block_size):
    """处理单个图像块并返回拼接后的图像块"""
    x, y, block = block_data
    best_match = find_best_match(block, sorted_small_images)
    best_match_resized = cv2.resize(best_match, (block_size, block_size))
    return x, y, best_match_resized


def create_mosaic(target_image_path, small_images_folder, block_size=10, output_image_path='result.jpg'):
    target_image = cv2.imread(target_image_path)
    height, width, _ = target_image.shape

    # 调整目标图像的尺寸
    target_height = (height // block_size) * block_size
    target_width = (width // block_size) * block_size
    target_image = cv2.resize(target_image, (target_width, target_height))

    small_images = []
    for filename in tqdm(os.listdir(small_images_folder)):
        img_path = os.path.join(small_images_folder, filename)
        if os.path.isfile(img_path):
            try:
                img = cv2.imread(img_path)
                img_resized = cv2.resize(img, (block_size, block_size))
                small_images.append(img_resized)
            except:
                print('Failed to add image:', img_path)

    # 对小图进行排序
    sorted_small_images = sort_small_images(small_images)

    blocks = get_image_blocks(target_image, block_size)
    mosaic_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    for block in tqdm(blocks):
        x, y, best_match_resized = process_block(block, sorted_small_images, block_size)
        mosaic_image[y:y + block_size, x:x + block_size] = best_match_resized

    cv2.imwrite(output_image_path, mosaic_image)


if __name__ == '__main__':
    target_image_path = './images/0HcOHnRoSfKxlsPjDeLC.jpg'  # 目标图像路径
    small_images_folder = './images'  # 存放小图的文件夹路径
    create_mosaic(target_image_path, small_images_folder)
