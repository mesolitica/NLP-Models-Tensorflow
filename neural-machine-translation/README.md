## How-to

1. Run [download-preprocess-dataset.ipynb](download-preprocess-dataset.ipynb) to download dataset and preprocessing.
2. Run any notebook using Jupyter Notebook.

## Accuracy, not sorted

Trainset to train, validation and test set to test. Based on test accuracy for 20 epochs. **Accuracy based on word positions**.

| name                                                       | accuracy |
|------------------------------------------------------------|----------|
| 1.basic-seq2seq-manual                                     | 0.103391 |
| 2.lstm-seq2seq-manual                                      | 0.118877 |
| 3.gru-seq2seq-manual                                       | 0.115032 |
| 4.basic-seq2seq-api-greedy                                 | 0.252812 |
| 5.lstm-seq2seq-api-greedy                                  | 0.330939 |
| 6.gru-seq2seq-greedy                                       | 0.312779 |
| 7.basic-birnn-seq2seq-manual                               | 0.125462 |
| 8.lstm-birnn-seq2seq-manual                                | 0.121065 |
| 9.gru-birnn-seq2seq-manual                                 | 0.119774 |
| 10.basic-birnn-seq2seq-greedy                              | 0.274987 |
| 11.lstm-birnn-seq2seq-greedy                               | 0.342469 |
| 12.gru-birnn-seq2seq-greedy                                | 0.325840 |
| 13.basic-seq2seq-luong                                     | 0.023959 |
| 14.lstm-seq2seq-luong                                      | 0.130840 |
| 15.gru-seq2seq-luong                                       | 0.073492 |
| 16.basic-seq2seq-bahdanau                                  | 0.915700 |
| 17.lstm-seq2seq-bahdanau                                   | 0.721833 |
| 18.gru-seq2seq-bahdanau                                    | 0.919218 |
| 19.lstm-birnn-seq2seq-luong                                | 0.918555 |
| 20.gru-birnn-seq2seq-luong                                 | 0.919445 |
| 21.lstm-birnn-seq2seq-bahdanau                             | 0.917655 |
| 22.gru-birnn-seq2seq-bahdanau                              | 0.920555 |
| 23.lstm-birnn-seq2seq-bahdanau-luong                       | 0.918182 |
| 24.gru-birnn-seq2seq-bahdanau-luong                        | 0.920045 |
| 25.lstm-seq2seq-greedy-luong                               | 0.364322 |
| 26.gru-seq2seq-greedy-luong                                | 0.627814 |
| 27.lstm-seq2seq-greedy-bahdanau                            | 0.378199 |
| 28.gru-seq2seq-greedy-bahdanau                             | 0.470696 |
| 29.lstm-seq2seq-beam                                       | 0.122135 |
| 30.gru-seq2seq-beam                                        | 0.163046 |
| 31.lstm-birnn-seq2seq-beam-luong                           | 0.171741 |
| 32.gru-birnn-seq2seq-beam-luong                            | 0.189787 |
| 33.lstm-birnn-seq2seq-luong-bahdanau-stack-beam            | 0.098961 |
| 34.gru-birnn-seq2seq-luong-bahdanau-stack-beam             | 0.091473 |
| 35.byte-net                                                | 1.022409 |
| 36.estimator                                               |          |
| 37.capsule-lstm-seq2seq-greedy                             |          |
| 38.capsule-lstm-seq2seq-luong-beam                         |          |
| 39.lstm-birnn-seq2seq-luong-bahdanau-stack-beam-dropout-l2 | 0.066305 |
| 40.dnc-seq2seq-bahdanau-greedy                             | 0.711184 |
| 41.lstm-birnn-seq2seq-beam-luongmonotic                    | 0.624756 |
| 42.lstm-birnn-seq2seq-beam-bahdanaumonotic                 | 0.624756 |
| 43.memory-network-basic                                    | 0.965700 |
| 44.memory-network-lstm                                     | 0.942591 |
| 45.attention-is-all-you-need                               | 0.170279 |
| 46.transformer-xl                                          | 0.114907 |
| 47.attention-is-all-you-need-beam-search                   | 0.158205 |
| 48.conv-encoder-conv-decoder                               | 0.462655 |
| 49.conv-encoder-lstm                                       | 0.438702 |
| 50.byte-net-greedy.ipynb                                   | 1.023528 |
| 51.gru-birnn-seq2seq-greedy-residual.ipynb                 | 0.561457 |
| 52.google-nmt.ipynb                                        | 0.675990 |
| 53.dilated-seq2seq.ipynb                                   | 1.023615 |
