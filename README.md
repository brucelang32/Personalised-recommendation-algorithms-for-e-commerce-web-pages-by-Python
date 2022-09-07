# LSTM-based-model-for-predicting-sales
I developed an LSTM model based on the tensorflow framework to predict the sales volume of electronic scales to help vendors better regulate their intake and reduce vegetable hoarding as well as waste.

# Problem Description
Vegetables in markets are likely to be in greater supply than demand, resulting in too many vegetables left over and extensive wasted. I imagine that if we could know the probable sales for the second day or the following days, the vendors would be able to reasonably regulate the amount of vegetables they need to sell everyday, reducing store and waste of vegetables.

We know that vegetable sales fluctuate mainly depending on seasons, weather, price and many other factors, and so far their impact on vegetable sales amount cannot be clearly defined. This is because sales prediction is highly non-linear, which requires prediction models to be able to deal with non-linearity and, as sales volume has a nature of time series, recurrent neural networks are suitable, for sales prediction.

# Model Select
The LSTM model is a special type of RNN model that solves the problem of RNNs not having long term memory.

# Input Parameter
Sales are influenced by time of year, weather, temperature and price. The input values are therefore defined as data, weather, temperature and price.


# Development
TensorFlow.js is an open source JavaScript-based library for training and implementing machine learning models. TensorFlow.js provides not only low-level machine learning building blocks, but also high-level Keras-like APIs for building neural networks. In my codes , the user can input weather and price and the Python script will give a reasonable prediction value.

# * The data cannot be uploaded due to confidentiality *
