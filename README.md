# NLP-Models
Solve NLP problems using Machine learning and Deep learning models

## Requirements
  * NumPy
  * scikit-learn
  * TensorFlow >= 1.2
  * matplotlib
  * scipy
  * gensim
  * Python 3.X

## Table of contents
  * [Text classification](https://github.com/huseinzol05/NLP-Models#text-classification)
  * [Chatbot](https://github.com/huseinzol05/NLP-Models#chatbot)
  * [Neural Machine Translation](https://github.com/huseinzol05/NLP-Models#neural-machine-translation-english-to-vietnam)
  * [Embedded](https://github.com/huseinzol05/NLP-Models#embedded)

## Models
### Text classification
#### Deep learning

1. Basic cell RNN
2. Basic cell RNN + Hinge
3. Basic cell RNN + Huber
4. Basic cell Bidirectional RNN
5. Basic cell Bidirectional RNN + Hinge
6. Basic cell Bidirectional RNN + Huber
7. LSTM cell RNN
8. LSTM cell RNN + Hinge
9. LSTM cell RNN + Huber
10. LSTM cell Bidirectional RNN
11. LSTM cell Bidirectional RNN + Huber
12. LSTM cell RNN + Dropout + L2
13. GRU cell RNN
14. GRU cell RNN + Hinge
15. GRU cell RNN + Huber
16. GRU cell Bidirectional RNN
17. GRU cell Bidirectional RNN + Hinge
18. GRU cell Bidirectional RNN + Huber
19. LSTM RNN + Conv2D
20. K-max Conv1d
21. LSTM RNN + Conv1D + Highway
22. LSTM RNN + Basic Attention
23. LSTM Dilated RNN
24. Layer-Norm LSTM cell RNN
25. Only Attention Neural Network
26. Multihead-Attention Neural Network
27. Neural Turing Machine
28. LSTM Seq2Seq
29. LSTM Seq2Seq + Luong Attention
30. LSTM Seq2Seq + Bahdanau Attention
31. LSTM Seq2Seq + Beam Decoder
32. LSTM Bidirectional Seq2Seq
33. Pointer Net
34. LSTM cell RNN + Bahdanau Attention
35. LSTM cell RNN + Luong Attention
36. LSTM cell RNN + Stack Bahdanau Luong Attention
37. LSTM cell Bidirectional RNN + backward Bahdanau + forward Luong
38. Bytenet
39. Fast-slow LSTM
40. Siamese Network
41. LSTM Seq2Seq + tf.estimator
42. Capsule layers + RNN LSTM
43. Capsule layers + LSTM Seq2Seq
44. Capsule layers + LSTM Bidirectional Seq2Seq
45. Nested LSTM
46. LSTM Seq2Seq + Highway

### Chatbot
#### Deep learning

1. Basic cell Seq2Seq-manual
2. LSTM Seq2Seq-manual
3. GRU Seq2Seq-manual
4. Basic cell Seq2Seq-API Greedy
5. LSTM Seq2Seq-API Greedy
6. GRU Seq2Seq-API Greedy
7. Basic cell Bidirectional Seq2Seq-manual
8. LSTM Bidirectional Seq2Seq-manual
9. GRU Bidirectional Seq2Seq-manual
10. Basic cell Bidirectional Seq2Seq-API Greedy
11. LSTM Bidirectional Seq2Seq-API Greedy
12. GRU Bidirectional Seq2Seq-API Greedy
13. Basic cell Seq2Seq-manual + Luong Attention
14. LSTM Seq2Seq-manual + Luong Attention
15. GRU Seq2Seq-manual + Luong Attention
16. Basic cell Seq2Seq-manual + Bahdanau Attention
17. LSTM Seq2Seq-manual + Bahdanau Attention
18. GRU Seq2Seq-manual + Bahdanau Attention
19. LSTM Bidirectional Seq2Seq-manual + Luong Attention
20. GRU Bidirectional Seq2Seq-manual + Luong Attention
21. LSTM Bidirectional Seq2Seq-manual + Bahdanau Attention
22. GRU Bidirectional Seq2Seq-manual + Bahdanau Attention
23. LSTM Bidirectional Seq2Seq-manual + backward Bahdanau + forward Luong
24. GRU Bidirectional Seq2Seq-manual + backward Bahdanau + forward Luong
25. LSTM Seq2Seq-API Greedy + Luong Attention
26. GRU Seq2Seq-API Greedy + Luong Attention
27. LSTM Seq2Seq-API Greedy + Bahdanau Attention
28. GRU Seq2Seq-API Greedy + Bahdanau Attention
29. LSTM Seq2Seq-API Beam Decoder
30. GRU Seq2Seq-API Beam Decoder
31. LSTM Bidirectional Seq2Seq-API + Luong Attention + Beam Decoder
32. GRU Bidirectional Seq2Seq-API + Luong Attention + Beam Decoder
33. LSTM Bidirectional Seq2Seq-API + backward Bahdanau + forward Luong + Stack Bahdanau Luong Attention + Beam Decoder
34. GRU Bidirectional Seq2Seq-API + backward Bahdanau + forward Luong + Stack Bahdanau Luong Attention + Beam Decoder
35. Bytenet
36. LSTM Seq2Seq + tf.estimator
37. Capsule layers + LSTM Seq2Seq-API Greedy
38. Capsule layers + LSTM Seq2Seq-API + Luong Attention + Beam Decoder

### Neural Machine Translation (English to Vietnam)

1. Basic cell Seq2Seq-manual
2. LSTM Seq2Seq-manual
3. GRU Seq2Seq-manual
4. Basic cell Seq2Seq-API Greedy
5. LSTM Seq2Seq-API Greedy
6. GRU Seq2Seq-API Greedy
7. Basic cell Bidirectional Seq2Seq-manual
8. LSTM Bidirectional Seq2Seq-manual
9. GRU Bidirectional Seq2Seq-manual
10. Basic cell Bidirectional Seq2Seq-API Greedy
11. LSTM Bidirectional Seq2Seq-API Greedy
12. GRU Bidirectional Seq2Seq-API Greedy
13. Basic cell Seq2Seq-manual + Luong Attention
14. LSTM Seq2Seq-manual + Luong Attention
15. GRU Seq2Seq-manual + Luong Attention
16. Basic cell Seq2Seq-manual + Bahdanau Attention
17. LSTM Seq2Seq-manual + Bahdanau Attention
18. GRU Seq2Seq-manual + Bahdanau Attention
19. LSTM Bidirectional Seq2Seq-manual + Luong Attention
20. GRU Bidirectional Seq2Seq-manual + Luong Attention
21. LSTM Bidirectional Seq2Seq-manual + Bahdanau Attention
22. GRU Bidirectional Seq2Seq-manual + Bahdanau Attention
23. LSTM Bidirectional Seq2Seq-manual + backward Bahdanau + forward Luong
24. GRU Bidirectional Seq2Seq-manual + backward Bahdanau + forward Luong
25. LSTM Seq2Seq-API Greedy + Luong Attention
26. GRU Seq2Seq-API Greedy + Luong Attention
27. LSTM Seq2Seq-API Greedy + Bahdanau Attention
28. GRU Seq2Seq-API Greedy + Bahdanau Attention
29. LSTM Seq2Seq-API Beam Decoder
30. GRU Seq2Seq-API Beam Decoder
31. LSTM Bidirectional Seq2Seq-API + Luong Attention + Beam Decoder
32. GRU Bidirectional Seq2Seq-API + Luong Attention + Beam Decoder
33. LSTM Bidirectional Seq2Seq-API + backward Bahdanau + forward Luong + Stack Bahdanau Luong Attention + Beam Decoder
34. GRU Bidirectional Seq2Seq-API + backward Bahdanau + forward Luong + Stack Bahdanau Luong Attention + Beam Decoder
35. Bytenet
36. LSTM Seq2Seq + tf.estimator

### Embedded

1. Word Vector using CBOW sample softmax
2. Word Vector using CBOW noise contrastive estimation
3. Word Vector using skipgram sample softmax
4. Word Vector using skipgram noise contrastive estimation
5. Lda2Vec Tensorflow
