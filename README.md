# Personalised-recommendation-algorithms-for-e-commerce-web-pages-by-Python
I developed an intelligent recommendation model using Collaborative Filtering (CF) techinque in order to help users of one law service e-commerce website quickly discover web pages of interest from the vast amount of information available.

# Background
The Lawtime.cn is a large Chinese e-commerce legal information platform, dedicated to providing users with a wealth of legal information and professional legal consulting services. As the scale of business enlarges, the number of visitors to their websites gradually increases, and with it the amount of data is also growing significantly. In order to save time of users and help them find information of interest quickly, Lawtime.cn posted a public project for a intelligent recommendation system.

# Dataset
The open-sourced dataset **data.sql** comes from a online chinese computer science community. It has SQL file format and provides user access logs of Lawtime.cn website. You can find it in the folder and you have to upload it into the SQL database first to access by Python.

# Algorithm Select
Collaborative filtering is a method of making automatic predictions (filtering) about the interests of a user by collecting preferences or taste information from many users. You can find more information about this technique from here: https://dl.acm.org/doi/abs/10.1145/138859.138867. (D. Goldberg, D. Nichols, B. M. Oki, and D. Terry, “Using collaborative filtering to weave an information tapestry,” Communications of ACM, vol. 35, no. 12, pp. 61–70, 1992.)  

# My model
My develoment is based on TensorFlow library for training and implementing machine learning models. TensorFlow provides not only low-level machine learning building blocks, but also high-level Keras APIs for building neural networks. In my codes, the user can input data, weather, temperature and price and then the Python script will give a reasonable prediction value.

# Results
You can find the results in the **Res.csv** where "IP address", "Viewed Webpages", "Recommended Webpages" and "Evaluation" are included. 
