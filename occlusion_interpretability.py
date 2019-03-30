import numpy as np
from tqdm import tqdm


class OcclusionInterpretability:

    def __init__(self, model):
        self.model = model

    def _occlude_image_2d(self, image, kernel, stride, batch=32):
        x_k, y_k = kernel
        x_img, y_img, channels = image.shape
        # Find padding amount
        x_pad = (kernel[0] - 1) // 2
        y_pad = (kernel[1] - 1) // 2
        padded_image = np.pad(image, [[x_pad, x_pad], [y_pad, y_pad], [0, 0]],
                              mode='constant', constant_values=0.0)

        # ((W - F * 2P)/S) + 1 and ((H - F * 2P)/S) + 1
        x_steps = int(((x_img - x_k + 2 * x_pad) / stride) + 1)
        y_steps = int(((y_img - y_k + 2 * y_pad) / stride) + 1)

        img_batch = []
        occluded_box = np.full((x_k, y_k, channels), 0.5)

        for y in range(y_steps):
            y_max = y + y_k
            for x in range(x_steps):
                x_max = x + x_k
                occluded_img = np.copy(padded_image)
                occluded_img[y:y_max, x:x_max, :] = occluded_box
                img_batch.append(occluded_img[y_pad:-y_pad, x_pad:-x_pad, :])
                if len(img_batch) >= batch:
                    yield np.array(img_batch)
                    img_batch = []
        yield np.array(img_batch)

    def convolution_occlusion(self, image, kernel=(5, 5), stride=1, batch=32):
        batch_iterator = self._occlude_image_2d(image, kernel, stride, batch)
        outputs = []
        total_iterations = image.shape[0] * image.shape[1]
        for i in tqdm(range(total_iterations)):
            output = self.model.predict(batch_iterator.__next__())
            outputs.append(output)
        return np.array(outputs)


if __name__ == "__main__":
    oc = OcclusionInterpretability(None)
    iterator = oc._occlude_image_2d(np.random.rand(10, 10, 1), (5, 5), 1)
