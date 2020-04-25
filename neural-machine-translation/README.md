## How-to

1. Run [download-preprocess-dataset.ipynb](download-preprocess-dataset.ipynb) to download dataset and preprocessing.
2. Run any notebook using Jupyter Notebook.

## Accuracy, not sorted

1. Trainset to train, validation and test set to test
2. Based on 20 epochs
3. Accuracy based on word positions
4. Some results are empty because the models are slow to train, still waiting for the results
5. Sort from shortest length to longest length and do bucketing from it will improve the accuracy by a lot.

| name                                                       | accuracy |
|------------------------------------------------------------|----------|
| 1.basic-seq2seq                                            | 0.103391 |
| 2.lstm-seq2seq                                             | 0.118877 |
| 3.gru-seq2seq                                              | 0.115032 |
| 4.basic-seq2seq-contrib-greedy                             | 0.252812 |
| 5.lstm-seq2seq-contrib-greedy                              | 0.330939 |
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
| 16.basic-seq2seq-bahdanau                                  | 0.132169 |
| 17.lstm-seq2seq-bahdanau                                   | 0.133821 |
| 18.gru-seq2seq-bahdanau                                    | 0.140176 |
| 19.basic-birnn-seq2seq-bahdanau                            | 0.138824 |
| 20.lstm-birnn-seq2seq-bahdanau                             | 0.131571 |
| 21.gru-birnn-seq2seq-bahdanau                              | 0.134661 |
| 22.basic-birnn-seq2seq-luong                               | 0.074942 |
| 23.lstm-birnn-seq2seq-luong                                | 0.132617 |
| 24.gru-birnn-seq2seq-luong                                 | 0.137604 |
| 25.lstm-seq2seq-contrib-greedy-luong                       | 0.455244 |
| 26.gru-seq2seq-contrib-greedy-luong                        | 0.081386 |
| 27.lstm-seq2seq-contrib-greedy-bahdanau                    | 0.438774 |
| 28.gru-seq2seq-contrib-greedy-bahdanau                     | 0.441251 |
| 29.lstm-seq2seq-contrib-beam-bahdanau                      | 0.244880 |
| 30.gru-seq2seq-contrib-beam-bahdanau                       | 0.222577 |
| 31.lstm-birnn-seq2seq-contrib-beam-luong                   | 0.241488 |
| 32.gru-birnn-seq2seq-contrib-beam-luong                    | 0.223249 |
| 33.lstm-birnn-seq2seq-contrib-luong-bahdanau-beam          |          |
| 34.gru-birnn-seq2seq-contrib-luong-bahdanau-beam           |          |
| 35.bytenet-greedy                                          |          |
| 36.capsule-lstm-seq2seq-contrib-greedy                     |          |
| 37.capsule-gru-seq2seq-contrib-greedy                      |          |
| 38.dnc-seq2seq-bahdanau-greedy                             |          |
| 39.dnc-seq2seq-luong-greedy                                |          |
| 40.lstm-birnn-seq2seq-beam-luongmonotic                    | 0.272306 |
| 41.lstm-birnn-seq2seq-beam-bahdanaumonotic                 | 0.263432 |
| 42.memory-network-lstm-seq2seq-contrib                     | 0.280245 |
| 43.attention-is-all-you-need-beam                          | 0.378041 |
| 44.conv-seq2seq                                            | 0.337291 |
| 45.conv-encoder-lstm-decoder                               | 0.329069 |
| 46.dilated-conv-seq2seq                                    | 0.331730 |
| 47.gru-birnn-seq2seq-greedy-residual                       | 0.343508 |
| 48.google-nmt                                              | 0.330886 |
| 49.bert-transformer-decoder-beam                           | 0.446938 |
| 50.xlnet-base-transformer-decoder-beam                     | 0.288339 |
| 51.evolved-transformer-tiny                                | 0.497402 |
