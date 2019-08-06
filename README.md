<p align="center">
    <a href="#readme">
        <img alt="logo" width="50%" src="tf-nlp.png">
    </a>
</p>
<p align="center">
  <a href="https://github.com/huseinzol05/NLP-Models-Tensorflow/blob/master/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  <a href="#"><img src="https://img.shields.io/badge/total%20models-299--models-blue.svg"></a>
  <a href="#"><img src="https://img.shields.io/badge/sentiments-75--models-success.svg"></a>
  <a href="#"><img src="https://img.shields.io/badge/chatbot-54--models-success.svg"></a>
  <a href="#"><img src="https://img.shields.io/badge/NMT-55--models-success.svg"></a>
</p>

---

**NLP-Models-Tensorflow**, Gathers machine learning and tensorflow deep learning models for NLP problems, **code simplify inside Jupyter Notebooks 100%**.

## Table of contents
  * [Text classification](https://github.com/huseinzol05/NLP-Models-Tensorflow#text-classification)
  * [Chatbot](https://github.com/huseinzol05/NLP-Models-Tensorflow#chatbot)
  * [Neural Machine Translation](https://github.com/huseinzol05/NLP-Models-Tensorflow#neural-machine-translation-english-to-vietnam)
  * [Embedded](https://github.com/huseinzol05/NLP-Models-Tensorflow#embedded)
  * [Entity-Tagging](https://github.com/huseinzol05/NLP-Models-Tensorflow#entity-tagging)
  * [POS-Tagging](https://github.com/huseinzol05/NLP-Models-Tensorflow#pos-tagging)
  * [Dependency-Parser](https://github.com/huseinzol05/NLP-Models-Tensorflow#dependency-parser)
  * [SQUAD Question-Answers](https://github.com/huseinzol05/NLP-Models-Tensorflow#squad-question-answers)
  * [Question-Answers](https://github.com/huseinzol05/NLP-Models-Tensorflow#question-answers)
  * [Abstractive Summarization](https://github.com/huseinzol05/NLP-Models-Tensorflow#abstractive-summarization)
  * [Extractive Summarization](https://github.com/huseinzol05/NLP-Models-Tensorflow#extractive-summarization)
  * [Stemming](https://github.com/huseinzol05/NLP-Models-Tensorflow#stemming)
  * [Generator](https://github.com/huseinzol05/NLP-Models-Tensorflow#generator)
  * [Topic Generator](https://github.com/huseinzol05/NLP-Models-Tensorflow#topic-generator)
  * [Language detection](https://github.com/huseinzol05/NLP-Models-Tensorflow#language-detection)
  * [OCR (optical character recognition)](https://github.com/huseinzol05/NLP-Models-Tensorflow#ocr-optical-character-recognition)
  * [Sentence-Pair classification](https://github.com/huseinzol05/NLP-Models-Tensorflow#sentence-pair)
  * [Speech to Text](https://github.com/huseinzol05/NLP-Models-Tensorflow#speech-to-text)
  * [Text to Speech](https://github.com/huseinzol05/NLP-Models-Tensorflow#text-to-speech)
  * [Old-to-Young Vocoder](https://github.com/huseinzol05/NLP-Models-Tensorflow#old-to-young-vocoder)
  * [Text Similarity](https://github.com/huseinzol05/NLP-Models-Tensorflow#text-similarity)
  * [Text Augmentation](https://github.com/huseinzol05/NLP-Models-Tensorflow#text-augmentation)
  * [Miscellaneous](https://github.com/huseinzol05/NLP-Models-Tensorflow#Miscellaneous)
  * [Attention](https://github.com/huseinzol05/NLP-Models-Tensorflow#attention)

## Objective

Original implementations are quite complex and not really beginner friendly. So I tried to simplify most of it. Also, there are tons of not-yet release papers implementation. So feel free to use it for your own research!

I will attached github repositories for models that I not implemented from scratch, basically I copy, paste and fix those code for deprecated issues.

## Tensorflow version

Tensorflow version 1.10 and above only, not included 2.X version.

## Contents

### [Text classification](text-classification)

Trained on [English sentiment dataset](https://github.com/huseinzol05/NLP-Models-Tensorflow/tree/master/text-classification/data).

1. Basic cell RNN
2. Bidirectional RNN
3. LSTM cell RNN
4. GRU cell RNN
5. LSTM RNN + Conv2D
6. K-max Conv1d
7. LSTM RNN + Conv1D + Highway
8. LSTM RNN with Attention
9. Neural Turing Machine
10. Bidirectional Transformers
11. Dynamic Memory Network
12. Residual Network using Atrous CNN + Bahdanau Attention
13. Transformer-XL
14. XL-net

<details><summary>Complete list (75 notebooks)</summary>

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
47. Triplet loss + LSTM
48. DNC (Differentiable Neural Computer)
49. ConvLSTM
50. Temporal Convd Net
51. Batch-all Triplet-loss + LSTM
52. Fast-text
53. Gated Convolution Network
54. Simple Recurrent Unit
55. LSTM Hierarchical Attention Network
56. Bidirectional Transformers
57. Dynamic Memory Network
58. Entity Network
59. End-to-End Memory Network
60. BOW-Chars Deep sparse Network
61. Residual Network using Atrous CNN
62. Residual Network using Atrous CNN + Bahdanau Attention
63. Deep pyramid CNN
64. Transformer-XL
65. Transfer learning GPT-2 345M
66. Quasi-RNN
67. Tacotron
68. Slice GRU
69. Slice GRU + Bahdanau
70. Wavenet
71. Transfer learning BERT base
72. Transfer learning XL-net Large
73. LSTM BiRNN global Max and average pooling
74. Transfer learning BERT base drop 6 layers
75. Transfer learning BERT large drop 12 layers

</details>

### [Chatbot](chatbot)

Trained on [Cornell Movie Dialog corpus](https://github.com/huseinzol05/NLP-Models-Tensorflow/blob/master/chatbot/dataset.tar.gz).

1. Seq2Seq-manual
2. Seq2Seq-API Greedy
3. Bidirectional Seq2Seq-manual
4. Bidirectional Seq2Seq-API Greedy
5. Bidirectional Seq2Seq-manual + backward Bahdanau + forward Luong
6. Bidirectional Seq2Seq-API + backward Bahdanau + forward Luong + Stack Bahdanau Luong Attention + Beam Decoder
7. Bytenet
8. Capsule layers + LSTM Seq2Seq-API + Luong Attention + Beam Decoder
9. End-to-End Memory Network
10. Attention is All you need
11. Transformer-XL + LSTM
12. GPT-2 + LSTM
13. Tacotron + Beam decoder

<details><summary>Complete list (54 notebooks)</summary>

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
39. LSTM Bidirectional Seq2Seq-API + backward Bahdanau + forward Luong + Stack Bahdanau Luong Attention + Beam Decoder + Dropout + L2
40. DNC Seq2Seq
41. LSTM Bidirectional Seq2Seq-API + Luong Monotic Attention + Beam Decoder
42. LSTM Bidirectional Seq2Seq-API + Bahdanau Monotic Attention + Beam Decoder
43. End-to-End Memory Network + Basic cell
44. End-to-End Memory Network + LSTM cell
45. Attention is all you need
46. Transformer-XL
47. Attention is all you need + Beam Search
48. Transformer-XL + LSTM
49. GPT-2 + LSTM
50. Fairseq
51. Conv-Encoder + LSTM
52. Tacotron + Greedy decoder
53. Tacotron + Beam decoder
54. Google NMT

</details>

### [Neural Machine Translation](neural-machine-translation)

Trained on [500 English-Vietnam](https://github.com/huseinzol05/NLP-Models-Tensorflow/blob/master/neural-machine-translation/vietnam-train).

1. Seq2Seq-manual
2. Seq2Seq-API Greedy
3. Bidirectional Seq2Seq-manual
4. Bidirectional Seq2Seq-API Greedy
5. Bidirectional Seq2Seq-manual + backward Bahdanau + forward Luong
6. Bidirectional Seq2Seq-API + backward Bahdanau + forward Luong + Stack Bahdanau Luong Attention + Beam Decoder
7. Bytenet
8. Capsule layers + LSTM Seq2Seq-API + Luong Attention + Beam Decoder
9. End-to-End Memory Network
10. Attention is All you need
11. BERT + Dilated Fairseq

<details><summary>Complete list (55 notebooks)</summary>

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
39. LSTM Bidirectional Seq2Seq-API + backward Bahdanau + forward Luong + Stack Bahdanau Luong Attention + Beam Decoder + Dropout + L2
40. DNC Seq2Seq
41. LSTM Bidirectional Seq2Seq-API + Luong Monotic Attention + Beam Decoder
42. LSTM Bidirectional Seq2Seq-API + Bahdanau Monotic Attention + Beam Decoder
43. End-to-End Memory Network + Basic cell
44. End-to-End Memory Network + LSTM cell
45. Attention is all you need
46. Transformer-XL
47. Attention is all you need + Beam Search
48. Fairseq
49. Conv-Encoder + LSTM
50. Bytenet Greedy
51. Residual GRU Bidirectional Seq2Seq-API Greedy
52. Google NMT
53. Dilated Seq2Seq
54. BERT Encoder + LSTM Luong Decoder
55. BERT Encoder + Dilated Fairseq

</details>

### [Embedded](embedded)

Trained on [English sentiment dataset](https://github.com/huseinzol05/NLP-Models-Tensorflow/tree/master/text-classification/data).

1. Word Vector using CBOW sample softmax
2. Word Vector using CBOW noise contrastive estimation
3. Word Vector using skipgram sample softmax
4. Word Vector using skipgram noise contrastive estimation
5. Lda2Vec Tensorflow
6. Supervised Embedded
7. Triplet-loss + LSTM
8. LSTM Auto-Encoder
9. Batch-All Triplet-loss LSTM
10. Fast-text
11. ELMO (biLM)
12. Triplet-loss + BERT

### [POS-Tagging](pos-tagging)

Trained on [CONLL POS](https://cogcomp.org/page/resource_view/81).

1. Bidirectional RNN + CRF, test accuracy 92%
2. Bidirectional RNN + Luong Attention + CRF, test accuracy 91%
3. Bidirectional RNN + Bahdanau Attention + CRF, test accuracy 91%
4. Char Ngrams + Bidirectional RNN + Bahdanau Attention + CRF, test accuracy 91%
5. Char Ngrams + Bidirectional RNN + Bahdanau Attention + CRF, test accuracy 91%
6. Char Ngrams + Residual Network + Bahdanau Attention + CRF, test accuracy 3%
7. Char Ngrams + Attention is you all Need + CRF, test accuracy 89%
8. BERT, test accuracy 99%

### [Entity-Tagging](entity-tagging)

Trained on [CONLL NER](https://cogcomp.org/page/resource_view/81).

1. Bidirectional RNN + CRF, test accuracy 96%
2. Bidirectional RNN + Luong Attention + CRF, test accuracy 93%
3. Bidirectional RNN + Bahdanau Attention + CRF, test accuracy 95%
4. Char Ngrams + Bidirectional RNN + Bahdanau Attention + CRF, test accuracy 96%
5. Char Ngrams + Bidirectional RNN + Bahdanau Attention + CRF, test accuracy 96%
6. Char Ngrams + Residual Network + Bahdanau Attention + CRF, test accuracy 69%
7. Char Ngrams + Attention is you all Need + CRF, test accuracy 90%
8. BERT, test accuracy 99%

### [Dependency-Parser](dependency-parser)

Trained on [CONLL English Dependency](https://github.com/huseinzol05/NLP-Models-Tensorflow/blob/master/dependency-parser/dev.conll.txt).

1. Bidirectional RNN + Bahdanau Attention + CRF
2. Bidirectional RNN + Luong Attention + CRF
3. Residual Network + Bahdanau Attention + CRF
4. Residual Network + Bahdanau Attention + Char Embedded + CRF
5. Attention is all you need + CRF

### [SQUAD Question-Answers](squad-qa)

Trained on [SQUAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/).

1. BERT,
```json
{"exact_match": 77.57805108798486, "f1": 86.18327335287402}
```

### [Question-Answers](question-answer)

Trained on [bAbI Dataset](https://research.fb.com/downloads/babi/).

1. End-to-End Memory Network + Basic cell
2. End-to-End Memory Network + GRU cell
3. End-to-End Memory Network + LSTM cell
4. Dynamic Memory

### [Stemming](stemming)

Trained on [English Lemmatization](https://github.com/huseinzol05/NLP-Models-Tensorflow/blob/master/stemming/lemmatization-en.txt).

1. LSTM + Seq2Seq + Beam
2. GRU + Seq2Seq + Beam
3. LSTM + BiRNN + Seq2Seq + Beam
4. GRU + BiRNN + Seq2Seq + Beam
5. DNC + Seq2Seq + Greedy
6. BiRNN + Bahdanau + Copynet

### [Abstractive Summarization](abstractive-summarization)

Trained on [India news](https://github.com/huseinzol05/NLP-Models-Tensorflow/tree/master/abstractive-summarization/dataset).

Accuracy based on 10 epochs only, calculated using word positions.

1. LSTM Seq2Seq using topic modelling, test accuracy 13.22%
2. LSTM Seq2Seq + Luong Attention using topic modelling, test accuracy 12.39%
3. LSTM Seq2Seq + Beam Decoder using topic modelling, test accuracy 10.67%
4. LSTM Bidirectional + Luong Attention + Beam Decoder using topic modelling, test accuracy 8.29%
5. Pointer-Generator + Bahdanau, https://github.com/xueyouluo/my_seq2seq, test accuracy 15.51%
6. Copynet, test accuracy 11.15%
7. Pointer-Generator + Luong, https://github.com/xueyouluo/my_seq2seq, test accuracy 16.51%
8. Dilated Seq2Seq, test accuracy 10.88%
9. Dilated Seq2Seq + Self Attention, test accuracy 11.54%
10. BERT + Dilated Fairseq, test accuracy 13.5%
11. self-attention + Pointer-Generator, test accuracy 4.34%
12. Dilated-Fairseq + Pointer-Generator, test accuracy 5.57%

### [Extractive Summarization](extractive-summarization)

Trained on [random books](https://github.com/huseinzol05/NLP-Models-Tensorflow/tree/master/extractive-summarization/books).

1. Skip-thought Vector
2. Residual Network using Atrous CNN
3. Residual Network using Atrous CNN + Bahdanau Attention

### [OCR (optical character recognition)](ocr)

1. CNN + LSTM RNN

### [Sentence-pair](sentence-pair)

Trained on [Cornell Movie--Dialogs Corpus](https://people.mpi-sws.org/~cristian/Cornell_Movie-Dialogs_Corpus.html)

1. BERT

### [Speech to Text](speech-to-text)

Trained on [Toronto speech dataset](https://tspace.library.utoronto.ca/handle/1807/24487).

1. Tacotron, https://github.com/Kyubyong/tacotron_asr
2. Bidirectional RNN + Greedy CTC
3. Bidirectional RNN + Beam CTC
4. Seq2Seq + Bahdanau Attention + Beam CTC
5. Seq2Seq + Luong Attention + Beam CTC
6. Bidirectional RNN + Attention + Beam CTC
7. Wavenet
8. CNN encoder + RNN decoder + Bahdanau Attention
9. CNN encoder + RNN decoder + Luong Attention
10. Dilation CNN + GRU Bidirectional
11. Deep speech 2
12. Pyramid Dilated CNN

### [Text to Speech](text-to-speech)

Trained on [Toronto speech dataset](https://tspace.library.utoronto.ca/handle/1807/24487).

1. Tacotron, https://github.com/Kyubyong/tacotron
2. Fairseq + Dilated CNN vocoder
3. Seq2Seq + Bahdanau Attention
4. Seq2Seq + Luong Attention
5. Dilated CNN + Monothonic Attention + Dilated CNN vocoder
6. Dilated CNN + Self Attention + Dilated CNN vocoder
7. Deep CNN + Monothonic Attention + Dilated CNN vocoder
8. Deep CNN + Self Attention + Dilated CNN vocoder

### [Old-to-Young Vocoder](vocoder)

Trained on [Toronto speech dataset](https://tspace.library.utoronto.ca/handle/1807/24487).

1. Dilated CNN

### [Generator](generator)

Trained on [Shakespeare dataset](https://github.com/huseinzol05/NLP-Models-Tensorflow/blob/master/generator/shakespeare.txt).

1. Character-wise RNN + LSTM
2. Character-wise RNN + Beam search
3. Character-wise RNN + LSTM + Embedding
4. Word-wise RNN + LSTM
5. Word-wise RNN + LSTM + Embedding
6. Character-wise + Seq2Seq + GRU
7. Word-wise + Seq2Seq + GRU
8. Character-wise RNN + LSTM + Bahdanau Attention
9. Character-wise RNN + LSTM + Luong Attention
10. Word-wise + Seq2Seq + GRU + Beam
11. Character-wise + Seq2Seq + GRU + Bahdanau Attention
12. Word-wise + Seq2Seq + GRU + Bahdanau Attention
13. Character-wise Dilated CNN + Beam search
14. Transformer + Beam search
15. Transformer XL + Beam search

### [Topic Generator](topic-generator)

Trained on [Malaysia news](https://github.com/huseinzol05/Malaya-Dataset/raw/master/news/news.zip).

1. TAT-LSTM
2. TAV-LSTM
3. MTA-LSTM
4. Dilated Fairseq

### [Language-detection](language-detection)

Trained on [Tatoeba dataset](http://downloads.tatoeba.org/exports/sentences.tar.bz2).

1. Fast-text Char N-Grams

### [Text Similarity](text-similarity)

Trained on [First Quora Dataset Release: Question Pairs](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs).

1. BiRNN + Contrastive loss, test accuracy 76.50%
2. Dilated CNN + Contrastive loss, test accuracy 72.98%
3. Transformer + Contrastive loss, test accuracy 73.48%
4. Dilated CNN + Cross entropy, test accuracy 72.27%
5. Transformer + Cross entropy, test accuracy 71.1%
6. Transfer learning BERT base + Cross entropy, test accuracy 90%

### [Text Augmentation](text-augmentation)

1. Pretrained Glove
2. GRU VAE-seq2seq-beam TF-probability
3. LSTM VAE-seq2seq-beam TF-probability
4. GRU VAE-seq2seq-beam + Bahdanau Attention TF-probability
5. VAE + Deterministic Bahdanau Attention, https://github.com/HareeshBahuleyan/tf-var-attention
6. VAE + VAE Bahdanau Attention, https://github.com/HareeshBahuleyan/tf-var-attention

### [Attention](attention)

1. Bahdanau
2. Luong
3. Hierarchical
4. Additive
5. Soft
6. Attention-over-Attention
7. Bahdanau API
8. Luong API

### [Miscellaneous](misc)

1. Attention heatmap on Bahdanau Attention
2. Attention heatmap on Luong Attention
3. BERT attention, https://github.com/hsm207/bert_attn_viz
4. XLNET attention

### [Not-deep-learning](not-deep-learning)

1. Markov chatbot
2. Decomposition summarization (3 notebooks)
