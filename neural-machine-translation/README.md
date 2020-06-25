## how-to

1. run [prepare-dataset.ipynb](prepare-dataset.ipynb).
2. run [prepare-bpe.ipynb](prepare-bpe.ipynb).
3. run [prepare-t2t.ipynb](prepare-t2t.ipynb).

## Notes

1. First 200k Trainset to train, validation and test set to test.
2. Based on 20 epochs.
3. Accuracy based on BLEU.
4. RNN and Transformer parameters are not consistent.

For RNN,

```python
size_layer = 512
num_layers = 2
```

For Transformer, we use BASE parameter from Tensor2Tensor.

Here we never tested what happened to RNN based models if we increase number of layers and size of layers same as Transformer BASE parameter.

5. Batch size not consistent, most of the models used 128 batch size.

## Accuracy, not sorted

| notebook                                                     | BLEU          |
|--------------------------------------------------------------|---------------|
| 1.basic-seq2seq.ipynb                                        | 6.319555e-05  |
| 2.lstm-seq2seq.ipynb                                         | 0.016924812   |
| 3.gru-seq2seq.ipynb                                          | 0.0094467895  |
| 4.basic-seq2seq-contrib-greedy.ipynb                         | 0.005418866   |
| 5.lstm-seq2seq-contrib-greedy.ipynb                          |               |
| 6.gru-seq2seq-contrib-greedy.ipynb                           | 0.051461186   |
| 7.basic-birnn-seq2seq.ipynb                                  | 6.319555e-05  |
| 8.lstm-birnn-seq2seq.ipynb                                   | 0.012854616   |
| 9.gru-birnn-seq2seq.ipynb                                    | 0.0095551545  |
| 10.basic-birnn-seq2seq-contrib-greedy.ipynb                  | 0.019748569   |
| 11.lstm-birnn-seq2seq-contrib-greedy.ipynb                   | 0.052993      |
| 12.gru-birnn-seq2seq-contrib-greedy.ipynb                    | 0.047413725   |
| 13.basic-seq2seq-luong.ipynb                                 | 8.97118e-05   |
| 14.lstm-seq2seq-luong.ipynb                                  | 0.053475615   |
| 15.gru-seq2seq-luong.ipynb                                   | 0.01888038    |
| 16.basic-seq2seq-bahdanau.ipynb                              | 0.00020161743 |
| 17.lstm-seq2seq-bahdanau.ipynb                               | 0.048261568   |
| 18.gru-seq2seq-bahdanau.ipynb                                | 0.025584696   |
| 19.basic-birnn-seq2seq-bahdanau.ipynb                        | 0.00020161743 |
| 20.lstm-birnn-seq2seq-bahdanau.ipynb                         | 0.054097746   |
| 21.gru-birnn-seq2seq-bahdanau.ipynb                          | 0.00020161743 |
| 22.basic-birnn-seq2seq-luong.ipynb                           |               |
| 23.lstm-birnn-seq2seq-luong.ipynb                            | 0.05320787    |
| 24.gru-birnn-seq2seq-luong.ipynb                             | 0.027758315   |
| 25.lstm-seq2seq-contrib-greedy-luong.ipynb                   | 0.15195806    |
| 26.gru-seq2seq-contrib-greedy-luong.ipynb                    | 0.101576895   |
| 27.lstm-seq2seq-contrib-greedy-bahdanau.ipynb                | 0.15275387    |
| 28.gru-seq2seq-contrib-greedy-bahdanau.ipynb                 | 0.13868862    |
| 29.lstm-seq2seq-contrib-beam-luong.ipynb                     | 0.17535137    |
| 30.gru-seq2seq-contrib-beam-luong.ipynb                      | 0.003980886   |
| 31.lstm-seq2seq-contrib-beam-bahdanau.ipynb                  | 0.17929372    |
| 32.gru-seq2seq-contrib-beam-bahdanau.ipynb                   | 0.1767827     |
| 33.lstm-birnn-seq2seq-contrib-beam-bahdanau.ipynb            | 0.19480321    |
| 34.lstm-birnn-seq2seq-contrib-beam-luong.ipynb               | 0.20042004    |
| 35.gru-birnn-seq2seq-contrib-beam-bahdanau.ipynb             | 0.1784567     |
| 36.gru-birnn-seq2seq-contrib-beam-luong.ipynb                | 0.0557322     |
| 37.lstm-birnn-seq2seq-contrib-beam-luongmonotonic.ipynb      | 0.06368613    |
| 38.gru-birnn-seq2seq-contrib-beam-luongmonotic.ipynb         | 0.06407658    |
| 39.lstm-birnn-seq2seq-contrib-beam-bahdanaumonotonic.ipynb   | 0.17586066    |
| 40.gru-birnn-seq2seq-contrib-beam-bahdanaumonotic.ipynb      | 0.065290846   |
| 41.residual-lstm-seq2seq-greedy-luong.ipynb                  | 0.1475228     |
| 42.residual-gru-seq2seq-greedy-luong.ipynb                   | 5.0574585e-05 |
| 43.residual-lstm-seq2seq-greedy-bahdanau.ipynb               | 0.15493448    |
| 44.residual-gru-seq2seq-greedy-bahdanau.ipynb                |               |
| 45.memory-network-lstm-decoder-greedy.ipynb                  |               |
| 46.google-nmt.ipynb                                          | 0.055380445   |
| 47.transformer-encoder-transformer-decoder.ipynb             | 0.17100729    |
| 48.transformer-encoder-lstm-decoder-greedy.ipynb             | 0.049064703   |
| 49.bertmultilanguage-encoder-bertmultilanguage-decoder.ipynb | 0.37003958    |
| 50.bertmultilanguage-encoder-lstm-decoder.ipynb              | 0.11384286    |
| 51.bertmultilanguage-encoder-transformer-decoder.ipynb       | 0.3941662     |
| 52.bertenglish-encoder-transformer-decoder.ipynb             | 0.23225775    |
| 53.transformer-t2t-2gpu.ipynb                                | 0.36773485    |