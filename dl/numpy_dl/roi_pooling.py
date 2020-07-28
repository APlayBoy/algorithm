import numpy as np


class ROIPooling2D:

    def __init__(self, pooled_height, pooled_width, spatial_scale=1.0):
        """

        :param pooled_height: 输出feature的height
        :param pooled_width: 输出feature的width
        :param spatial_scale:
        """
        self.pooled_height, self.pooled_width = pooled_height, pooled_width
        self.spatial_scale = spatial_scale

    def forward(self, inputs, rois):
        """
        :param inputs: feature map
        :param rois: anchor左上角和右下角的坐标的集合
        :return:
        """
        n_images, channels, height, width = inputs.shape
        n_rois = rois.shape[0]
        y = np.zeros((n_images, n_rois, channels, self.pooled_height,
                      self.pooled_width), dtype=np.float32)
        spatial_scale = self.spatial_scale
        pooled_height, pooled_width = self.pooled_height, self.pooled_width
        for roi_n in range(n_rois):
            roi_start_w = round(rois[roi_n][1] * spatial_scale)
            roi_start_h = round(rois[roi_n][2] * spatial_scale)
            roi_end_w = round(rois[roi_n][3] * spatial_scale)
            roi_end_h = round(rois[roi_n][4] * spatial_scale)

            # Force malformed ROIs to be 1x1
            roi_width = max(roi_end_w - roi_start_w + 1, 1)
            roi_height = max(roi_end_h - roi_start_h + 1, 1)

            bin_size_h = float(roi_height) / pooled_height
            bin_size_w = float(roi_width) / pooled_width
            for pi in range(n_images):
                for ph in range(self.pooled_height):
                    for pw in range(self.pooled_width):
                        hstart = np.floor(bin_size_h * ph)
                        wstart = np.floor(bin_size_w * pw)
                        hend = np.floor(bin_size_h * (ph + 1))
                        wend = np.floor(bin_size_w * (pw + 1))
                        hstart = int(min(max(hstart + roi_start_h, 0), height))
                        hend = int(min(max(hend + roi_start_h, 0), height))
                        wstart = int(min(max(wstart + roi_start_w, 0), width))
                        wend = int(min(max(wend + roi_start_w, 0), width))
                        is_empty = (hend <= hstart) or (wend <= wstart)
                        if not is_empty:
                            y[pi, roi_n, :, ph, pw] = np.max(inputs[pi, :, hstart:hend, wstart:wend], axis=(2, 3))
        return y