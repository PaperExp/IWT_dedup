from phash import phash

import cv2
import hashlib
import math
import numpy as np
import time

img_block_size_power = 5    # power of image block length(byte): 32 = 2^5 
blk_l = 1 << img_block_size_power

base_side_len = 8
base_size = base_side_len * base_side_len * 1   # base size = base point number * point size
# diff size = (block point number - base point number) * point size
diff_size = ((1 << img_block_size_power) * (1 << img_block_size_power) - base_side_len * base_side_len) * 1
hash_size = base_side_len * base_side_len >> 3  # hash is int, one point -> one bit, byte = bit / 8

class fuzzy_dedup():
    def __init__(self) -> None:
        # arguments
        self.base_dict = {}     # base dictionary
        self.diff_dict = {}     # diff dictionary
        self.file_cnt = 0       # upload file count
        self.err_cnt = 0        # fuzzy dedup error count
        # storage overhead
        self.base_strg_oh = 0   # base storage overhead
        self.diff_strg_oh = 0   # diff storage overhead
        # store time
        self.up_time = 0.0
        self.down_time = 0.0
        self.diff_dedup_time = 0.0

    @staticmethod
    def dvd_img(img : cv2.Mat, img_type):
        img_bs = []

        # point high and width
        h, w = img.shape
        # complete block high and width
        b_h, b_w = (h >> img_block_size_power), (w >> img_block_size_power)
        # remainder point high and width
        rmd_h, rmd_w = h & ((1 << img_block_size_power) - 1), w & ((1 << img_block_size_power) - 1)

        # every block-row
        for i in range(b_h):
            row_bs = []
            start_p_i = i << img_block_size_power
            # every block-col
            for j in range(b_w):
                start_p_j = j << img_block_size_power
                imgb = np.zeros((blk_l, blk_l), img_type)   # image block
                # add point to image block
                for bi in range(blk_l):
                    for bj in range(blk_l):
                        # print((start_p_i, bi, start_p_j, bj))
                        imgb[bi, bj] = img[bi + start_p_i, bj + start_p_j]
                row_bs.append(imgb)

            # last remainder block in one row
            if rmd_w != 0:
                start_p_j = b_w << img_block_size_power
                imgb = np.zeros((blk_l, rmd_w), img_type)
                # add point to image block
                for bi in range(blk_l):
                    for bj in range(rmd_w):
                        imgb[bi, bj] = img[bi + start_p_i, bj + start_p_j]
                row_bs.append(imgb)

            img_bs.append(row_bs)

        # last row remainder block
        if rmd_h != 0:
            row_bs = []
            start_p_i = b_h << img_block_size_power
            for j in range(b_w):
                start_p_j = j << img_block_size_power
                imgb = np.zeros((rmd_h, blk_l), img_type)   #image block
                # add point to image block
                for bi in range(rmd_h):
                    for bj in range(blk_l):
                        imgb[bi, bj] = img[bi + start_p_i, bj + start_p_j]
                row_bs.append(imgb)

            # last remainder block in one row
            if rmd_w != 0:
                start_p_j = rmd_w << img_block_size_power
                imgb = np.zeros((rmd_h, rmd_w), img_type)
                # add point to image block
                for bi in range(rmd_h):
                    for bj in range(rmd_w):
                        imgb[bi, bj] = img[bi + start_p_i, bj + start_p_j]
                row_bs.append(imgb)

            img_bs.append(row_bs)

        return img_bs

    @staticmethod
    def get_base_size(h : int, w : int, ratio : float):
        bh, bw = math.ceil(h * ratio), math.ceil(w * ratio)
        if bh == 0:
            bh = 2
        if bw == 0:
            bw = 2

        if bh & 1 != 0:
            bh += 1
        if bw & 1 != 0:
            bw += 1
        return bh, bw

    @staticmethod
    def dvd_img_size(img, img_type, bh, bw):
        img_bs = []

        # point high and width
        h, w = img.shape
        # complete block high and width
        b_h, b_w = h // bh, w // bw
        # remainder point high and width
        rmd_h, rmd_w = h % bh, w % bw

        # every block-row
        for i in range(b_h):
            row_bs = []
            start_p_i = i * bh
            # every block-col
            for j in range(b_w):
                start_p_j = j * bw
                imgb = np.zeros((bh, bw), img_type)   # image block
                # add point to image block
                for bi in range(bh):
                    for bj in range(bw):
                        # print((start_p_i, bi, start_p_j, bj))
                        imgb[bi, bj] = img[bi + start_p_i, bj + start_p_j]
                row_bs.append(imgb)

            # last remainder block in one row
            if rmd_w != 0:
                start_p_j = b_w * bw
                imgb = np.zeros((bh, rmd_w), img_type)
                # add point to image block
                for bi in range(bh):
                    for bj in range(rmd_w):
                        imgb[bi, bj] = img[bi + start_p_i, bj + start_p_j]
                row_bs.append(imgb)

            img_bs.append(row_bs)

        # last row remainder block
        if rmd_h != 0:
            row_bs = []
            start_p_i = b_h * bh
            for j in range(b_w):
                start_p_j = j * bw
                imgb = np.zeros((rmd_h, bw), img_type)   #image block
                # add point to image block
                for bi in range(rmd_h):
                    for bj in range(bw):
                        imgb[bi, bj] = img[bi + start_p_i, bj + start_p_j]
                row_bs.append(imgb)

            # last remainder block in one row
            if rmd_w != 0:
                start_p_j = rmd_w * bw
                imgb = np.zeros((rmd_h, rmd_w), img_type)
                # add point to image block
                for bi in range(rmd_h):
                    for bj in range(rmd_w):
                        imgb[bi, bj] = img[bi + start_p_i, bj + start_p_j]
                row_bs.append(imgb)

            img_bs.append(row_bs)

        return img_bs

    @staticmethod
    def is_fuzzy_exist(hash : int, tar_dict : dict, dist : int) -> bool:
        i = 0
        exist = False
        while i <= dist and not exist:
            if hash - i in tar_dict.keys():
                exist = True
            if hash + i in tar_dict.keys():
                exist = True
            i += 1
        return exist

    @staticmethod
    def get_img_strg(img):
        h, w = img.shape
        return h * w
    
    @staticmethod
    def get_jpg_compress_size(img):
        params = [cv2.IMWRITE_JPEG_QUALITY, 99]
        msg = cv2.imencode('.jpg', img, params)[1]
        msg = (np.array(msg)).tobytes()
        return len(msg)
    
    @staticmethod
    def b_phash(val, avrg):
        if val > avrg:
            return 1
        else:
            return 0

    @staticmethod
    def phash(img):
        h, w = img.shape
        avrg = np.mean(img)

        # storage hash
        hash = 0
        i, j, d = 0, 0, 1
        while i != h - 1 or j != w - 1:
            # down
            if d == 0:
                while i != h - 1 and j != 0:
                    hash = (hash << 1) | fuzzy_dedup.b_phash(img[i, j], avrg)
                    i, j = i + 1, j - 1
                # last point
                hash = (hash << 1) | fuzzy_dedup.b_phash(img[i, j], avrg)
                # next base point
                if i == h - 1:
                    j += 1
                else:
                    i += 1
                d == 1  # next to up
            # up
            else:
                while i != 0 and j != w - 1:
                    hash = (hash << 1) | fuzzy_dedup.b_phash(img[i, j], avrg)
                    i, j = i - 1, j + 1
                # last point
                hash = (hash << 1) | fuzzy_dedup.b_phash(img[i, j], avrg)
                # next base point
                if j == w - 1:
                    i += 1
                else:
                    j += 1
                d == 0  # next to down
        # the right and down point
        hash = (hash << 1) | fuzzy_dedup.b_phash(img[i, j], avrg)
        return hash

    @staticmethod
    def dvd_base_diff(img, bh : int, bw : int):
        img_dct = cv2.dct(np.float32(img))
        h, w = img.shape
        base_h = min(h, bh)
        base_w = min(w, bw)

        # divide base
        base = np.zeros((base_h, base_w), np.float32)
        for i in range(base_h):
            for j in range(base_w):
                base[i, j] = img_dct[i, j]
        # base = cv2.idct(base)

        # divide diff
        diff = []

        diff1 = np.zeros((base_h, w - base_w), np.float32)
        for i in range(base_h):
            for j in range(base_w, w):
                diff1[i, j - base_w] = img_dct[i, j]
        # diff1 = cv2.idct(diff1)
        diff.append(diff1)

        diff2 = np.zeros((h - base_h, base_w), np.float32)
        for i in range(base_h, h):
            for j in range(base_w):
                diff2[i - base_h, j] = img_dct[i, j]
        # diff2 = cv2.idct(diff2)
        diff.append(diff2)

        diff3 = np.zeros((h - base_h, w - base_w), np.float32)
        for i in range(base_h, h):
            for j in range(base_w, w):
                diff3[i - base_h, j - base_w] = img_dct[i, j]
        # diff3 = cv2.idct(diff3)
        diff.append(diff3)

        return [base, diff]

    def upload(self, image_path):
        img = cv2.imread(image_path, flags=2)
        h, w = img.shape

        # upload process
        self.up_time -= time.time()
        bh, bw = fuzzy_dedup.get_base_size(h, w, 1/80)
        img_dct = cv2.dct(np.float32(img))
        # transport base and diff, not divide to blocks
        (base, diff) = fuzzy_dedup.dvd_base_diff(img, bh, bw)

        # storage base and diff, blocks to fixed size as base high and base width
        blks = fuzzy_dedup.dvd_img_size(img_dct, np.float32, bh, bw)

        ph = fuzzy_dedup.phash(base)
        if not(ph in self.base_dict.keys()):
            self.base_dict[ph] = 1
            self.base_strg_oh = self.base_strg_oh + fuzzy_dedup.get_img_strg(base)
        else:
            self.err_cnt += 1   # dedup, but modify img should not dedup
        self.up_time += time.time()

        # diff deduplication process
        self.diff_dedup_time -= time.time()
        # storage diff
        for i in range(len(blks)):
            for j in range(len(blks[i])):
                if i == 0 and j == 0:
                    continue
                dh = fuzzy_dedup.phash(blks[i][j])
                if not fuzzy_dedup.is_fuzzy_exist(dh, self.diff_dict, 5):
                    self.diff_dict[dh] = 1
                    self.diff_strg_oh = self.diff_strg_oh + fuzzy_dedup.get_img_strg(blks[i][j])
        self.file_cnt = self.file_cnt + 1
        self.diff_dedup_time += time.time()

    def download(self, image_path):
        img = cv2.imread(image_path, flags=2)
        h, w = img.shape
        bh, bw = fuzzy_dedup.get_base_size(h, w, 1/80)
        (base, diff) = fuzzy_dedup.dvd_base_diff(img[0:h, 0:w], bh, bw)

        self.down_time -= time.time()
        rvt_img = np.zeros((h, w), np.float32)
        for i in range(bh):
            for j in range(bw):
                rvt_img[i, j] = base[i, j]

        diff1_dct = diff[0]
        for i in range(bh):
            for j in range(w - bw):
                rvt_img[i, bw + j] = diff1_dct[i, j]

        diff2_dct = diff[1]
        for i in range(h - bh):
            for j in range(bw):
                rvt_img[bh + i, j] = diff2_dct[i, j]

        diff3_dct = diff[2]
        for i in range(h - bh):
            for j in range(w - bw):
                rvt_img[bh + i, bw + j] = diff3_dct[i, j]

        rvt_img = cv2.idct(rvt_img)
        self.down_time += time.time()
