// feito utilizando collab
from glob import glob
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from google.colab.patches import cv2_imshow

from IPython.display import HTML, display

def set_css():
  display(HTML('''
  <style>
    pre {
        white-space: pre-wrap;
    }
  </style>
  '''))
get_ipython().events.register('pre_run_cell', set_css)

nomes_imgs = glob(os.path.join(os.getcwd(), '/content/drive/MyDrive/VC/vc1', '*.jpg'))
fonte = cv2.FONT_HERSHEY_SIMPLEX

print(f'Imagens a serem analisadas: {nomes_imgs}')
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Lê todas as imagens na pasta 'imgs' e as plota
for nome in nomes_imgs:
    img = cv2.imread(nome)
    B, G, R = cv2.split(img)

    # Aplicação de filtros
    img_blt = cv2.bilateralFilter(G, 1, 90, 90)
    img_blr = cv2.blur(img_blt, (5, 5))

    # Binarização (retorna somente o canal verde binarizado)
    img_thr = cv2.threshold(img_blr, 190, 255, cv2.THRESH_BINARY)[1]

    # Dilatação
    img_dlt = cv2.dilate(img_thr, np.ones((4, 4), np.uint8), iterations=1)

plt.figure(figsize=(16, 8))

plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')

plt.subplot(2, 3, 2)
plt.imshow(G, cmap='gray')
plt.title('Canal verde')

plt.subplot(2, 3, 3)
plt.imshow(img_blt, cmap='gray')
plt.title('Filtro Bilateral')

plt.subplot(2, 3, 4)
plt.imshow(img_blr, cmap='gray')
plt.title('Filtro Blur')

plt.subplot(2, 3, 5)
plt.imshow(img_thr, cmap='gray')
plt.title('Binarizada')

plt.subplot(2, 3, 6)
plt.imshow(img_dlt, cmap='gray')
plt.title('Dilatação')

plt.show()

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
from glob import glob
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from google.colab.patches import cv2_imshow

from IPython.display import HTML, display

def set_css():
  display(HTML('''
  <style>
    pre {
        white-space: pre-wrap;
    }
  </style>
  '''))
get_ipython().events.register('pre_run_cell', set_css)

nomes_imgs = glob(os.path.join(os.getcwd(), '/content/drive/MyDrive/VC/vc2', '*.jpg'))
fonte = cv2.FONT_HERSHEY_SIMPLEX

print(f'Imagens a serem analisadas: {nomes_imgs}')

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Lê todas as imagens na pasta 'imgs' e as plota
for nome in nomes_imgs:
    img = cv2.imread(nome)
    B, G, R = cv2.split(img)

    # Aplicação de filtros
    img_blt = cv2.bilateralFilter(G, 1, 90, 90)
    img_blr = cv2.blur(img_blt, (5, 5))

    # Binarização (retorna somente o canal verde binarizado)
    img_thr = cv2.threshold(img_blr, 190, 255, cv2.THRESH_BINARY)[1]

    # Dilatação
    img_dlt = cv2.dilate(img_thr, np.ones((4, 4), np.uint8), iterations=1)

plt.figure(figsize=(16, 8))

plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')

plt.subplot(2, 3, 2)
plt.imshow(G, cmap='gray')
plt.title('Canal verde')

plt.subplot(2, 3, 3)
plt.imshow(img_blt, cmap='gray')
plt.title('Filtro Bilateral')

plt.subplot(2, 3, 4)
plt.imshow(img_blr, cmap='gray')
plt.title('Filtro Blur')

plt.subplot(2, 3, 5)
plt.imshow(img_thr, cmap='gray')
plt.title('Binarizada')

plt.subplot(2, 3, 6)
plt.imshow(img_dlt, cmap='gray')
plt.title('Dilatação')

plt.show()
