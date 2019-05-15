## How-to

1. Unzip [dataset.tar.gz](dataset.tar.gz)

2. Run any notebook using Jupyter Notebook.

## Accuracy, not sorted

Based on 20 epochs accuracy. The results will be different on different dataset. Trained on a GTX 960, 4GB VRAM.

| name                                                       | accuracy |
|------------------------------------------------------------|----------|
| 1.basic-seq2seq-manual                                     | 0.816000 |
| 2.lstm-seq2seq-manual                                      | 0.735000 |
| 3.gru-seq2seq-manual                                       | 0.846833 |
| 4.basic-seq2seq-api-greedy                                 | 1.009119 |
| 5.lstm-seq2seq-api-greedy                                  | 0.984596 |
| 6.gru-seq2seq-greedy                                       | 1.008869 |
| 7.basic-birnn-seq2seq-manual                               | 0.990333 |
| 8.lstm-birnn-seq2seq-manual                                | 0.732833 |
| 9.gru-birnn-seq2seq-manual                                 | 0.936667 |
| 10.basic-birnn-seq2seq-greedy                              | 1.009586 |
| 11.lstm-birnn-seq2seq-greedy                               | 0.991938 |
| 12.gru-birnn-seq2seq-greedy                                | 1.008791 |
| 13.basic-seq2seq-luong                                     | 0.821167 |
| 14.lstm-seq2seq-luong                                      | 0.723167 |
| 15.gru-seq2seq-luong                                       | 0.751667 |
| 16.basic-seq2seq-bahdanau                                  | 0.811833 |
| 17.lstm-seq2seq-bahdanau                                   | 0.721833 |
| 18.gru-seq2seq-bahdanau                                    | 0.728167 |
| 19.lstm-birnn-seq2seq-luong                                | 0.728500 |
| 20.gru-birnn-seq2seq-luong                                 | 0.743833 |
| 21.lstm-birnn-seq2seq-bahdanau                             | 0.718833 |
| 22.gru-birnn-seq2seq-bahdanau                              | 0.746667 |
| 23.lstm-birnn-seq2seq-bahdanau-luong                       | 0.721000 |
| 24.gru-birnn-seq2seq-bahdanau-luong                        | 0.747667 |
| 25.lstm-seq2seq-greedy-luong                               | 0.974864 |
| 26.gru-seq2seq-greedy-luong                                | 0.999175 |
| 27.lstm-seq2seq-greedy-bahdanau                            | 0.987874 |
| 28.gru-seq2seq-greedy-bahdanau                             | 1.000434 |
| 29.lstm-seq2seq-beam                                       | 0.874802 |
| 30.gru-seq2seq-beam                                        | 0.905397 |
| 31.lstm-birnn-seq2seq-beam-luong                           | 0.913772 |
| 32.gru-birnn-seq2seq-beam-luong                            | 0.856824 |
| 33.lstm-birnn-seq2seq-luong-bahdanau-stack-beam            | 0.732801 |
| 34.gru-birnn-seq2seq-luong-bahdanau-stack-beam             | 0.756537 |
| 35.byte-net                                                | 0.877510 |
| 36.estimator                                               |          |
| 37.capsule-lstm-seq2seq-greedy                             | 0.655007 |
| 38.capsule-lstm-seq2seq-luong-beam                         | 0.275569 |
| 39.lstm-birnn-seq2seq-luong-bahdanau-stack-beam-dropout-l2 | 0.312999 |
| 40.dnc-seq2seq-bahdanau-greedy                             | 0.962712 |
| 41.lstm-birnn-seq2seq-beam-luongmonotic                    | 0.917333 |
| 42.lstm-birnn-seq2seq-beam-bahdanaumonotic                 | 0.929333 |
| 43.memory-network-basic                                    | 0.945333 |
| 44.memory-network-lstm                                     | 0.900000 |
| 45.attention-is-all-you-need                               | 0.704549 |
| 46.transformer-xl                                          | 0.874486 |
| 47.attention-is-all-you-need-beam-search                   | 0.836433 |
| 48.transformer-xl-lstm                                     | 0.826571 |
| 49.gpt-2-lstm                                              | 0.645157 |
| 50.conv-encoder-conv-decoder                               | 0.518504 |
| 51.conv-encoder-lstm                                       | 0.924609 |
| 52.tacotron-greedy                                         | 0.876267 |
| 53.tacotron-beam                                           | 0.855140 |
| 54.google-nmt                                              | 1.006089 |
