# Sarcasm detection

Using tf/keras and nltk to determine whether a news headline is sarcastic or not

## Running the project
The dataset can be found at https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection/kernels
with credit to https://rishabhmisra.github.io/publications/

To run with anaconda, download and extract the dataset(place the folder in the same directory)

Make an anaconda environment and run:

    conda install --file requirements.txt

Then to train and evaluate the model, run:

    python main.py


## Implementation

The dataset contains the following information:

| headline | article_link | is_sarcastic |
| -------- | ------------ | ------------ |
| ...      |  ...         | ...          |

My model only considers how the news headline affects likelihood of sarcasm, but this could be
expanded by using the url as another input, or scraping text from the actual article.


### Encoding the headlines

To start off with, we have a headline such as:

    "the fascinating case for eating lab-grown meat"

Firstly, I used nltk to remove the stopwords - common words such as "a, the, it":

    "fascinating case eating lab-grown meat"

Next, I used nltk's lemmatization tool - reducing words to some stem form in English

    "fascinate case eat lab-grown meat"

The words can then be converted to a sequence of numbers in a dictionary mapping.

We can use the mapping to convert headlines into a sequence of numbers and pad them to the same length. The resulting vectors
can be fed into the neural network.


### Results

Validation and training results with a batch size of 1000 over 10 epochs, and a split of
30/70 for validation/training data:

![alt text](https://github.com/rowanho/sarcasm_detection/blob/master/graphs/val.png "validation")

The model achieved around 80% accuracy on the final test data
