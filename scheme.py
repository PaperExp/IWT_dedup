from phash import phash
from iwt import image_to_iwt, iwt_to_image
from Crypto.Cipher import Salsa20

# import cv2
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
        # store time
        self.up_time = 0.0
        self.down_time = 0.0
        self.diff_dedup_time = 0.0

    def __encrypt(self, base : np.ndarray):
        bbase = base.tobytes()
        hbase = hashlib.sha256(bbase).digest()
        cipher = Salsa20.new(key=hbase)
        ciper_base = cipher.encrypt(bbase)
        return hbase, ciper_base, cipher.nonce
    
    def __decrypt(self, cipher_base : bytes, key : bytes, nonce : bytes, h : int, w : int):
        cipher = Salsa20.new(key=key, nonce=nonce)
        bbase = cipher.decrypt(cipher_base)
        return np.frombuffer(bbase, dtype=np.int64).reshape((h, w))

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
        # request upload
        start_time = time.time()

        approximation, (horizontal, vertical, diagonal) = image_to_iwt(image_path)
        key, cipher_base, nonce = self.__encrypt(approximation)
        h, w = approximation.shape
        req_hash = hashlib.sha256(key).hexdigest()
        if req_hash not in self.base_dic:
            self.base_dic[req_hash] = (cipher_base, h, w)

        end_time = time.time()
        self.up_time += (end_time - start_time)

        # fuzzy deduplication for diff
        start_time = time.time()

        hp = self.add_to_diff(horizontal)
        vp = self.add_to_diff(vertical)
        dp = self.add_to_diff(diagonal)
        if req_hash not in self.file_dic:
            self.file_dic[req_hash] = (key, nonce, hp, vp, dp)

        end_time = time.time()
        self.diff_dedup_time += (end_time - start_time)

        return req_hash

    def download(self, req_hash):
        start_time = time.time()
        if req_hash not in self.base_dic:
            return None
        cipher_base, h, w = self.base_dic[req_hash]
        key, nonce, hp, vp, dp = self.file_dic[req_hash]
        approximation = self.__decrypt(cipher_base, key, nonce, h, w)
        horizontal = self.diff_dic[hp]
        vertical = self.diff_dic[vp]
        diagonal = self.diff_dic[dp]
        img = iwt_to_image(approximation, (horizontal, vertical, diagonal))
        end_time = time.time()
        self.down_time += (end_time - start_time)
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