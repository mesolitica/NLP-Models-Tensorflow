## How-to

1. Make sure `data` folder in the same directory of the notebooks.

2. Run any notebook using Jupyter Notebook.

## Score and average time taken per epoch, not sorted

Based on 20% validation. The results will be different on different dataset. Trained on a GTX 960, 4GB VRAM.

| name                                 | accuracy | time taken (s) |
|--------------------------------------|----------|----------------|
| 1. basic-rnn                         | 0.68     | 1.3219         |
| 2. basic-rnn-hinge                   | 0.65     | 1.2455         |
| 3. basic-rnn-huber                   | 0.68     | 1.2468         |
| 4. basic-rnn-bidirectional           | 0.71     | 3.8174         |
| 5. basic-rnn-bidirectional-hinge     | 0.68     | 2.5127         |
| 6. basic-rnn-bidirectional-huber     | 0.63     | 3.5095         |
| 7. lstm-rnn                          | 0.73     | 2.69683        |
| 8. lstm-rnn-hinge                    | 0.72     | 8.2088         |
| 9. lstm-rnn-huber                    | 0.73     | 10.1754        |
| 10. lstm-rnn-bidirectional           | 0.71     | 11.0388        |
| 11. lstm-rnn-bidirectional-huber     | 0.71     | 5.5258         |
| 12. lstm-rnn-dropout-l2              | 0.74     | 3.2420         |
| 13. gru-rnn                          | 0.72     | 3.16123        |
| 14. gru-rnn-hinge                    | 0.72     | 6.71951        |
| 15. gru-rnn-huber                    | 0.70     | 7.93373        |
| 16. gru-rnn-bidirectional            | 0.73     | 2.91590        |
| 17. gru-rnn-bidirectional-hinge      | 0.72     | 5.66385        |
| 18. gru-rnn-bidirectional-huber      | 0.70     | 18.01133       |
| 19. lstm-cnn-rnn                     | 0.65     | 4.42849        |
| 20. kmax-cnn                         | 0.73     | 18.89667       |
| 21. lstm-cnn-rnn-highway             | 0.68     | 3.23122        |
| 22. lstm-rnn-attention               | 0.75     | 13.97496       |
| 23. dilated-rnn-lstm                 | 0.25     | 24.54002       |
| 24. lnlstm-rnn                       | 0.68     | 24.86363       |
| 25. only-attention                   | 0.74     | 2.63291        |
| 26. multihead-attention              | 0.69     | 9.033228       |
| 27. neural-turing-machine            |          |                |
| 28. lstm-seq2seq                     | 0.72     | 9.63291        |
| 29. lstm-seq2seq-luong               |          |                |
| 30. lstm-seq2seq-bahdanau            |          |                |
| 31. lstm-seq2seq-beam                |          |                |
| 32. lstm-seq2seq-birnn               |          |                |
| 33. pointer-net                      |          |                |
| 34. lstm-rnn-bahdanau                | 0.71     | 9.81993        |
| 35. lstm-rnn-luong                   | 0.66     | 27.73932       |
| 36. lstm-rnn-bahdanau-luong          | 0.69     | 36.97628       |
| 37. lstm-birnn-bahdanau-luong        | 0.70     | 38.86009       |
| 38. bytenet                          |          |                |
| 39. fast-slow-lstm                   |          |                |
| 40. siamese-network                  | 0.52     | 7.13535        |
| 41. estimator                        |          |                |
| 42. capsule-rnn-lstm                 |          |                |
| 43. capsule-seq2seq-lstm             |          |                |
| 44. capsule-birrn-seq2seq-lstm       |          |                |
| 45. nested-lstm                      |          |                |
| 46. lstm-seq2seq-highway             |          |                |
| 47. triplet-loss-lstm                | 0.50     |                |
| 48. dnc                              | 0.68     | 85.98529       |
| 49. convlstm                         | 0.69     | 2.66726        |
| 50. temporalconvd                    | 0.66     | 11.90590       |
| 51. batch-all-triplet-loss-lstm      | 0.70     |                |
| 52. fast-text                        | 0.76     | 0.49499        |
| 53. gated-convolution-network        | 0.67     | 3.37712        |
| 54. simple-recurrent-units           | 0.65     | 3.12624        |
| 55. lstm-han                         | 0.50     | 3.47965        |
| 56. bert                             | 0.73     | 6.31015        |
| 57. dynamic-memory-network           | 0.71     | 3.25820        |
| 58. entity-network                   | 0.74     | 1.10458        |
| 59. memory-network                   | 0.58     | 1.157306       |
| 60. char-sparse                      | 0.76     | 2.350096       |
| 61. residual-network                 | 0.72     | 9.557085       |
| 62. residual-network-bahdanau        | 0.71     | 11.53799       |
| 63. deep-pyramid-cnn                 | 0.68     | 6.980528       |
| 64. transformer-xl                   | 0.51     | 38.66338       |
| 65. transfer-learning-gpt2           | 0.79     | 178.0716       |
| 66. quasi-rnn                        | 0.66     | 166.1456       |
| 67. tacotron                         | 0.74     | 360.5551       |
| 68. slice-gru                        | 0.72     | 10.140633      |
| 69. slice-gru-bahdanau               | 0.70     | 20.247409      |
| 70. wavenet                          | 0.59     | 101.293274     |
| 71. transfer-learning-bert           | 0.81     | 887.590460     |
| 72. transfer-learning-xlnet-large    | 0.846    | 340.7679       |
| 73. lstm-birnn-max-avg               | 0.7552   | 9.35624        |
| 74. transfer-learning-bert-base-6    | 0.7655   | 494.169        |
| 75. transfer-learning-bert-large-12  | 0.80     | 1365.30        |
| 76. transfer-learning-xlnet-base     | 0.820441 | 240.262        |
