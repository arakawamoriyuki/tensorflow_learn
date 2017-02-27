- bazelでbuildしなくていい。(できなかった)

```
python lm_1b/lm_1b_eval.py --mode sample \
                           --prefix "I love that I" \
                           --pbtxt data/graph-2016-09-10.pbtxt \
                           --vocab_file data/vocab-2016-09-10.txt  \
                           --ckpt 'data/ckpt-*'

python lm_1b/lm_1b_eval.py --mode eval \
                           --pbtxt data/graph-2016-09-10.pbtxt \
                           --vocab_file data/vocab-2016-09-10.txt  \
                           --input_data data/news.en.heldout-00000-of-00050 \
                           --ckpt 'data/ckpt-*'


python lm_1b/lm_1b_eval.py --mode dump_emb \
                           --pbtxt data/graph-2016-09-10.pbtxt \
                           --vocab_file data/vocab-2016-09-10.txt  \
                           --ckpt 'data/ckpt-*' \
                           --save_dir output


python lm_1b/lm_1b_eval.py --mode dump_lstm_emb \
                           --pbtxt data/graph-2016-09-10.pbtxt \
                           --vocab_file data/vocab-2016-09-10.txt \
                           --ckpt 'data/ckpt-*' \
                           --sentence "I love who I am ." \
                           --save_dir output
```

```
import numpy as np
import matplotlib.pyplot as plt

vector_x = np.arange(-3, 3, 0.1)
vector_y = np.sin(vector_x)
plt.plot(vector_x, vector_y)
plt.show()

metrics = np.load('output/embeddings_char_cnn.npy')
metrics = np.load('output/embeddings_softmax.npy') # 5G 重くて開け無い(そもそも開くファイルじゃ無い？)
metrics = np.load('output/lstm_emb_step_0.npy')

plt.hist(metrics)
plt.imshow(metrics)

plt.show()
```