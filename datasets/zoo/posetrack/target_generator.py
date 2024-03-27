import numpy as np
import math

class HeatmapGenerator():
    def __init__(self, output_res):
        self.output_res = output_res

    def gaussian2D(self, shape, sigma=1.):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def __call__(self, keypoints, bboxes):
        num_keypoints = keypoints.shape[1]

        hms = np.zeros((num_keypoints, self.output_res, self.output_res), dtype=np.float32)
        mask = np.ones((num_keypoints, 1, 1), dtype=np.float32)

        for kpt, bbox in zip(keypoints, bboxes):
            bbox = np.clip(bbox, 0, self.output_res - 1)
            h = np.sqrt(np.power(bbox[2, 0]-bbox[0, 0], 2) + np.power(bbox[2, 1] - bbox[0, 1], 2))
            w = np.sqrt(np.power(bbox[1, 0]-bbox[0, 0], 2) + np.power(bbox[1, 1] - bbox[0, 1], 2))
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius))
            diameter = 2 * radius + 1
            gaussian = self.gaussian2D((diameter, diameter), sigma=diameter / 6)
            height, width = self.output_res, self.output_res

            for idx, pt in enumerate(kpt):
                if pt[2] > 0:
                    x, y = int(pt[0]), int(pt[1])
                    if x < 0 or y < 0 or x >= self.output_res or y >= self.output_res: continue

                    left, right = min(x, radius), min(width - x, radius + 1)
                    top, bottom = min(y, radius), min(height - y, radius + 1)

                    masked_heatmap = hms[idx][y - top:y + bottom, x - left:x + right]
                    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
                    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
                        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)
                else:
                    mask[idx] = 0.0
        return hms, mask

def gaussian_radius(det_size, min_overlap=0.7):
  height, width = det_size

  a1  = 1
  b1  = (height + width)
  c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  r1  = (b1 + sq1) / 2

  a2  = 4
  b2  = 2 * (height + width)
  c2  = (1 - min_overlap) * width * height
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  r2  = (b2 + sq2) / 2

  a3  = 4 * min_overlap
  b3  = -2 * min_overlap * (height + width)
  c3  = (min_overlap - 1) * width * height
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3  = (b3 + sq3) / 2
  return min(r1, r2, r3)
