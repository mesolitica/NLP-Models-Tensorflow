## How-to

1. Unzip [dataset.tar.gz](dataset.tar.gz)

2. Run any notebooks.

## Accuracy, not sorted

Based on 20 epochs accuracy. The results will be different on different dataset. Trained on a GTX 960, 4GB VRAM.

| name                                                       | accuracy |
|------------------------------------------------------------|----------|
| 1.basic-seq2seq-manual                                     | 0.433333 |
| 2.lstm-seq2seq-manual                                      | 0.279335 |
| 3.gru-seq2seq-manual                                       | 0.689382 |
| 4.basic-seq2seq-api-greedy                                 | 0.146505 |
| 5.lstm-seq2seq-api-greedy                                  |          |
| 6.gru-seq2seq-greedy                                       | 0.839214 |
| 7.basic-birnn-seq2seq-manual                               | 0.944960 |
| 8.lstm-birnn-seq2seq-manual                                | 0.315659 |
| 9. lstm-rnn-huber                                          | 0.839214 |
| 10.basic-birnn-seq2seq-greedy                              | 0.939953 |
| 11.lstm-birnn-seq2seq-greedy                               | 0.906821 |
| 12.gru-birnn-seq2seq-greedy                                | 0.941129 |
| 13.basic-seq2seq-luong                                     | 0.374496 |
| 14.lstm-seq2seq-luong                                      | 0.265491 |
| 15.gru-seq2seq-luong                                       | 0.335551 |
| 16.basic-seq2seq-bahdanau                                  | 0.392944 |
| 17.lstm-seq2seq-bahdanau                                   | 0.256452 |
| 18.gru-seq2seq-bahdanau                                    | 0.269388 |
| 19.lstm-birnn-seq2seq-luong                                | 0.267204 |
| 20.gru-birnn-seq2seq-luong                                 | 0.301277 |
| 21.lstm-birnn-seq2seq-bahdanau                             | 0.261626 |
| 22.gru-birnn-seq2seq-bahdanau                              | 0.305981 |
| 23.lstm-birnn-seq2seq-bahdanau-luong                       | 0.267776 |
| 24.gru-birnn-seq2seq-bahdanau-luong                        | 0.364281 |
| 25.lstm-seq2seq-greedy-luong                               | 0.906250 |
| 26.gru-seq2seq-greedy-luong                                | 0.937601 |
| 27.lstm-seq2seq-greedy-bahdanau                            | 0.917977 |
| 28.gru-seq2seq-greedy-bahdanau                             | 0.936828 |
| 29.lstm-seq2seq-beam                                       | 0.797581 |
| 30.gru-seq2seq-beam                                        | 0.930712 |
| 31.lstm-birnn-seq2seq-beam-luong                           | 0.767137 |
| 32.gru-birnn-seq2seq-beam-luong                            | 0.812601 |
| 33.lstm-birnn-seq2seq-luong-bahdanau-stack-beam            | 0.739987 |
| 34.gru-birnn-seq2seq-luong-bahdanau-stack-beam             | 0.732796 |
| 35.byte-net                                                |          |
| 36.estimator                                               |          |
| 37.capsule-lstm-seq2seq-greedy                             |          |
| 38.capsule-lstm-seq2seq-luong-beam                         |          |
| 39.lstm-birnn-seq2seq-luong-bahdanau-stack-beam-dropout-l2 | 0.850773 |
| 40.dnc-seq2seq-bahdanau-greedy                             | 0.898824 |
| 41.lstm-birnn-seq2seq-beam-luongmonotic                    | 0.680444 |
| 42.lstm-birnn-seq2seq-beam-bahdanaumonotic                 | 0.748656 |
