# Integer Sequence Learning
https://www.kaggle.com/c/integer-sequence-learning

## Competition Details
7\. You read that correctly. That's the start to a real integer sequence, the powers of primes. Want something easier? How about the next number in 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55? If you answered 89, you may enjoy this challenge. Your computer may find it considerably less enjoyable.

The On-Line Encyclopedia of Integer Sequences is a 50+ year effort by mathematicians the world over to catalog sequences of integers. If it has a pattern, it's probably in the OEIS, and probably described with amazing detail. This competition challenges you create a machine learning algorithm capable of guessing the next number in an integer sequence. While this sounds like pattern recognition in its most basic form, a quick look at the data will convince you this is anything but basic!

## Evaluation
This competition is evaluated on accuracy of your predictions (the percentage of sequences where you predict the next number correctly).

## Data
This dataset contains the majority of the integer sequences from the OEIS. It is split into a training set, where you are given the full sequence, and a test set, where we have removed the last number from the sequence. The task is to predict this removed integer.

Note that some sequences may have identical beginnings (or even be identical altogether). We have not removed these from the dataset.

## Discussion
https://www.kaggle.com/c/integer-sequence-learning/discussion/21671
