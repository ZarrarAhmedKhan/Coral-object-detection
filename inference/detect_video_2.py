# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Example using PyCoral to detect objects in a given image.

To run this code, you must attach an Edge TPU attached to the host and
install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
device setup instructions, see coral.ai/docs/setup.

Example usage:
```
bash examples/install_requirements.sh detect_image.py

python3 examples/detect_image.py \
  --model test_data/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite \
  --labels test_data/coco_labels.txt \
  --input test_data/grace_hopper.bmp \
  --output ${HOME}/grace_hopper_processed.bmp
```
"""

import argparse
import time

from PIL import Image
from PIL import ImageDraw

import cv2
import numpy

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter


def draw_objects(draw, objs, labels):
  """Draws the bounding box and label for each object."""
  for obj in objs:
    bbox = obj.bbox
    draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                   outline='red')
    draw.text((bbox.xmin + 10, bbox.ymin + 10),
              '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
              fill='red')


def main():

  cap = cv2.VideoCapture(0)
  if not cap.isOpened():
    raise IOError("Cannot open webcam")

  labels = read_label_file('../mask-labels.txt')
  interpreter = make_interpreter('../efficientdet2-lite-mask.tflite')
  interpreter.allocate_tensors()
  inference_size = input_size(interpreter)
  counter = 0
  while True:
    ret, image = cap.read()
    if not ret:
      break
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # image = Image.open(image)
    _, scale = common.set_resized_input(interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))

    print('----INFERENCE TIME----')
    print('Note: The first inference is slow because it includes',
          'loading the model into Edge TPU memory.')
    if counter == 0:
          interpreter.invoke()
          objs = detect.get_objects(interpreter, 0.6, scale)



    counter = 1
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    objs = detect.get_objects(interpreter, 0.6, scale)
    fps = 1000 / (inference_time * 1000)
    print('%.2f fps' %  fps )
    print('%.2f ms' %  (inference_time * 1000) )
    print('-------')

    print('-------RESULTS--------')
    if not objs:
      print('No objects detected')

    for obj in objs:
      print(labels.get(obj.id, obj.id))
      print('  id:    ', obj.id)
      print('  score: ', obj.score)
      print('  bbox:  ', obj.bbox)

    draw_objects(ImageDraw.Draw(image), objs, labels)
    open_cv_image = numpy.array(image)
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    cv2.imshow("image", open_cv_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  cap.release()
  cv2.destroyAllWindows()

  # if args.output:
  #   image = image.convert('RGB')
    # draw_objects(ImageDraw.Draw(image), objs, labels)
  #   image.save(args.output)
  #   image.show()


if __name__ == '__main__':
  main()
