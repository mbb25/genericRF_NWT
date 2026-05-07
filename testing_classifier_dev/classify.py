import joblib
import numpy as np
from scipy.ndimage import zoom

class ChunkClassifier:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = joblib.load(self.model_path)
    
    def derive_features(self, rgb_chunk, chm_chunk):
        # Map from feature names to channels
        R = 0
        G = 1
        B = 2
        chm = 3
        rc = 4
        gc = 5
        bc = 6
        rc_d_gc = 7
        rc_p_gc = 8
        ExG = 9
        ExR = 10
        ExB = 11
        ExGmExR = 12
        Ikaw = 13
        MGRVI = 14
        GLI = 15
        Y = 16
        L = 17
        z_score_Y = 18
        z_score_L = 19

        _, height, width = rgb_chunk.shape
        chunk = np.zeros((20, height, width))

        # Copy basic features
        chunk[R] = rgb_chunk[R]
        chunk[G] = rgb_chunk[G]
        chunk[B] = rgb_chunk[B]
        chunk[chm] = chm_chunk

        eps = 1e-9
        chunk[rc] = chunk[R] / (chunk[R] + chunk[G] + chunk[B] + eps)
        chunk[gc] = chunk[G] / (chunk[R] + chunk[G] + chunk[B] + eps)

        # blue chromaticity
        chunk[bc] = 1-(chunk[rc]+chunk[gc])

        # ration rc/gc
        chunk[rc_d_gc] = chunk[rc]/(chunk[gc] + eps)

        # sum rc+gc
        chunk[rc_p_gc] = chunk[rc]+chunk[gc]

        # excess indecies
        chunk[ExG] = 2*chunk[G] - chunk[R] - chunk[B]

        chunk[ExR] = 1.4*chunk[R] - chunk[G]

        chunk[ExB] = 1.4*chunk[B] - chunk[G]

        # index ExGmExR
        chunk[ExGmExR] = chunk[ExG]-chunk[ExR]

        # index Ikaw - Kawashima Index
        chunk[Ikaw] = (chunk[R]-chunk[B])/(chunk[R]+chunk[B]+eps)

        # MGRVI - modified green red vegetation index
        chunk[MGRVI] = ((chunk[G])**2 - (chunk[R])**2)/((chunk[G])**2 + (chunk[R])**2 + eps)

        # GLI - green leaf index
        chunk[GLI] = (2*chunk[G] - chunk[R] - chunk[B])/(2*chunk[G] + chunk[R] + chunk[B] + eps)

        # luminance Y
        sRGBtoLin = lambda x: np.where(x <= 0.04045, x/12.92, (pow(((x + 0.055)/1.055), 2.4)))
        chunk[Y] = 0.2126*sRGBtoLin(chunk[R]) + 0.7152*sRGBtoLin(chunk[G]) + 0.0722*sRGBtoLin(chunk[B])

        # perceived lightness L
        YtoLstar = lambda y: np.where(y <= (216/24389), y*(24389/27), (pow(y, (1/3)) * 116 - 16))
        chunk[L] = YtoLstar(chunk[Y])

        # z_scores for Y and L
        check = np.where(chunk[chm] >= 0)
        valid_Y_np = chunk[Y][check]
        Y_mean = valid_Y_np.mean()
        Y_std = valid_Y_np.std()
        del valid_Y_np

        valid_L_np = chunk[L][check]
        L_mean = valid_L_np.mean()
        L_std = valid_L_np.std()
        del valid_L_np

        chunk[z_score_Y] = (chunk[Y] - Y_mean)/(Y_std+eps)
        chunk[z_score_L] = (chunk[L] - L_mean)/(L_std+eps)

        return chunk


    def classify_chunk(self, chunk):
        _, h, w = chunk.shape
        output = np.zeros((1, h, w))
        py, px = np.where(chunk[3] >= 0)
        features = chunk[:, py, px].T
        output[:, py, px] = self.model.predict(features)
        return output
    

    def chm_from_dsm(self, dsm_chunk):
        bins = np.array([0.0, 3.0, 10.0])
        return np.digitize(dsm_chunk, bins)


    def process(self, rgb_chunk, dsm_chunk):
        check = np.where(dsm_chunk >= 0)
        if len(check[0]) == 0:
            print(f'  Empty chunk. Skipping process...')
            return np.zeros(dsm_chunk.shape)
        
        chm_chunk = self.chm_from_dsm(dsm_chunk)
        feature_chunk = self.derive_features(rgb_chunk, chm_chunk)
        return self.classify_chunk(feature_chunk)

    