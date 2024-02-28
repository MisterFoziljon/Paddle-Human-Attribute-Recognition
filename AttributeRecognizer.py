import os
import yaml
import glob
from functools import reduce

import cv2
import numpy as np
import math
import paddle
from paddle.inference import Config
from paddle.inference import create_predictor

from preprocess import preprocess, Resize, NormalizeImage, Permute, PadStride, LetterBoxResize, WarpAffine
from infer import Detector

from PIL import Image
paddle.enable_static()

class HumanAttributeRecognizer(Detector):
    def __init__(self, model_dir):
        super(HumanAttributeRecognizer, self).__init__(model_dir=model_dir, device="CPU", batch_size=1, threshold=0.5)

    def postprocess(self, inputs, result):
        im_results = result['output']
        result = {
            "Jinsi:":None,
            "Yoshi":None, 
            "Yo'nalishi":None,
            "Ko'zoynak":None,
            "Bosh kiyim":None,
            "Narsa ko'tarib olgan":None,
            "Sumka":None,
            "Yuqori kiyim":None,
            "Past kiyim":None,
            "Etik":None
        }
        
        age_list = ['18 dan kichik', "18-60 oralig'i", "60 dan katta"]
        direct_list = ['Oldinga', 'Yonga', 'Orqaga']
        bag_list = ["Qo'l sumkasi", 'Yelkaga osiladigan sumka', 'Ryugzag']
        upper_list = ['UpperStride', 'UpperLogo', 'UpperPlaid', 'UpperSplice']
        lower_list = ['LowerStripe', 'LowerPattern', 'LongCoat', 'Shim', "Sho'rtik", 'Yubka']
        
        glasses_threshold = 0.3
        hold_threshold = 0.6
        batch_res = []
        
        for res in im_results:
            res = res.tolist()
            label_res = []
             
            gender = 'Ayol' if res[22] > self.threshold else 'Erkak'
            result["Jinsi:"] = gender
            
            age = age_list[np.argmax(res[19:22])]
            result["Yoshi"] = age
             
            direction = direct_list[np.argmax(res[23:])]
            result["Yo'nalishi"] = direction
            
            if res[1] > glasses_threshold:
                result["Ko'zoynak"] = "Bor"
            else:
                result["Ko'zoynak"] = "Yo'q"
            
            if res[0] > self.threshold:
                result["Bosh kiyim"] = "Bor"
            else:
                result["Bosh kiyim"] = "Yo'q"
            
            if res[18] > hold_threshold:
                result["Narsa ko'tarib olgan"] = "Ha"
            else:
                result["Narsa ko'tarib olgan"] = "Yo'q"

            bag = bag_list[np.argmax(res[15:18])]
            bag_score = res[15 + np.argmax(res[15:18])]
            bag_label = bag if bag_score > self.threshold else "Yo'q"
            result["Sumka"] = bag_label
            
            sleeve = 'Yengi uzun' if res[3] > res[2] else 'Yengi kalta'
            upper_label = '{}'.format(sleeve)
            
            upper_res = res[4:8]
            if np.max(upper_res) > self.threshold:
                upper_label += ' {}'.format(upper_list[np.argmax(upper_res)])
            result["Yuqori kiyim"] = upper_label
            
            lower_res = res[8:14]
            lower_label = '' 
            has_lower = False
            for i, l in enumerate(lower_res):
                if l > self.threshold:
                    lower_label += '{}'.format(lower_list[i])+" "
                    has_lower = True
            if not has_lower:
                lower_label += '{}'.format(lower_list[np.argmax(lower_res)])
            result["Past kiyim"] = lower_label
            
            shoe = 'Kiygan' if res[14] > self.threshold else "Kiymagan"
            result["Etik"] = shoe

        return result

    def predict(self, repeats=1):
        for i in range(repeats):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            output_tensor = self.predictor.get_output_handle(output_names[0])
            np_output = output_tensor.copy_to_cpu()
        result = dict(output=np_output)
        return result

    def __call__(self, image_list, visual=True):
        batch_loop_cnt = math.ceil(float(len(image_list)) / self.batch_size)
        results = []
        for i in range(batch_loop_cnt):
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, len(image_list))
            batch_image_list = image_list[start_index:end_index]

            # preprocess
            self.det_times.preprocess_time_s.start()
            inputs = self.preprocess(batch_image_list)
            self.det_times.preprocess_time_s.end()

            # model prediction
            self.det_times.inference_time_s.start()
            result = self.predict()
            self.det_times.inference_time_s.end()

            # postprocess
            self.det_times.postprocess_time_s.start()
            result = self.postprocess(inputs, result)
            self.det_times.postprocess_time_s.end()
            
            self.det_times.img_num += len(batch_image_list)
            results.append(result)
        return results