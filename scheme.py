from phash import phash
from iwt import image_to_iwt, iwt_to_image
from Crypto.Cipher import Salsa20

import cv2
import hashlib
import numpy as np
import time

class iwt_dedup:
    def __init__(self) -> None:
        # hamming distance threshold
        self.t = 1
        # index dictionary to map base and diff
        self.base_dic = {}
        self.diff_dic = {}
        # store all messages in files, the key is req_hash, and value is (key, hp, vp, dp)
        self.file_dic = {}

    def __encrypt(self, base : np.ndarray):
        bbase = base.tobytes()
        hbase = hashlib.sha256(bbase).digest()
        cipher = Salsa20.new(key=hbase)
        return hbase, cipher.encrypt(bbase)
    
    def __decrypt(self, cipher_base : bytes, key : bytes, h : int, w : int):
        cipher = Salsa20.new(key=key)
        return np.frombuffer(cipher.decrypt(cipher_base), dtype=np.uint8).reshape((h, w))

    def add_to_diff(self, diff : np.ndarray):
        diff_phash = phash(diff)
        # compare diff_phash with all phash in diff_dic
        # if the hamming distance is less than t, we can say that the diff is already in the diff_dic
        # so we can return the index of the diff in diff_dic
        for k, v in self.diff_dic.items():
            if np.count_nonzero(diff_phash ^ k) <= self.t:
                return k
            
        # if the diff is not in the diff_dic, we can add it to the diff_dic
        # and return the index of the diff in diff_dic
        self.diff_dic[diff_phash] = diff
        return diff_phash

    def upload(self, image_path):
        approximation, (horizontal, vertical, diagonal) = image_to_iwt(image_path)
        key, cipher_base = self.__encrypt(approximation)
        h, w = approximation.shape
        req_hash = hashlib.sha256(key).hexdigest()
        if req_hash not in self.base_dic:
            self.base_dic[req_hash] = (cipher_base, h, w)
        
        hp = self.add_to_diff(horizontal)
        vp = self.add_to_diff(vertical)
        dp = self.add_to_diff(diagonal)
        if req_hash not in self.file_dic:
            self.file_dic[req_hash] = (key, hp, vp, dp)

        return req_hash

    def download(self, req_hash):
        if req_hash not in self.base_dic:
            return None
        cipher_base, h, w = self.base_dic[req_hash]
        key, hp, vp, dp = self.file_dic[req_hash]
        approximation = self.__decrypt(cipher_base, key, h, w)
        horizontal = self.diff_dic[hp]
        vertical = self.diff_dic[vp]
        diagonal = self.diff_dic[dp]
        img = iwt_to_image(approximation, (horizontal, vertical, diagonal))
        # img = cv2.convertScaleAbs(img)
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

if __name__ == '__main__':
    serv = iwt_dedup()
    start_time = time.time()
    req_hash = serv.upload('./bill/medical-record-org2.jpeg')
    end_time = time.time()
    print('Upload time: %f' % (end_time - start_time))
    start_time = time.time()
    serv.download(req_hash)
    end_time = time.time()
    print('Download time: %f' % (end_time - start_time))