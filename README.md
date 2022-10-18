# Personalised-recommendation-algorithms-for-e-commerce-web-pages-by-Python
I developed an intelligent recommendation model using Collaborative Filtering (CF) techinque in order to help users of one law service e-commerce website quickly discover web pages of interest from the vast amount of information available.

# Background
The Lawtime.cn is a large Chinese e-commerce legal information platform, dedicated to providing users with a wealth of legal information and professional legal consulting services. As the scale of business enlarges, the number of visitors to their websites gradually increases, and with it the amount of data is also growing significantly. In order to save time of users and help them find information of interest quickly, Lawtime.cn posted a public project for a intelligent recommendation system.

# Dataset
The open-sourced dataset **data.sql** comes from a online chinese computer science community. It has SQL file format and provides user access logs of Lawtime.cn website. The dataset is larger than 500 MB so it cannot be uploaed here. You can find it here:[data](https://blog.csdn.net/duan_zhihua/article/details/82871000?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166610116116800186516733%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=166610116116800186516733&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-8-82871000-null-null.142^v59^pc_search_tree,201^v3^add_ask&utm_term=%E6%B3%95%E5%BE%8B%E5%BF%AB%E8%BD%A6&spm=1018.2226.3001.4187)

# Algorithm Select
Collaborative filtering is a method of making automatic predictions (filtering) about the interests of a user by collecting preferences or taste information from many users. You can find more information about this technique from here: https://dl.acm.org/doi/abs/10.1145/138859.138867. (D. Goldberg, D. Nichols, B. M. Oki, and D. Terry, “Using collaborative filtering to weave an information tapestry,” Communications of ACM, vol. 35, no. 12, pp. 61–70, 1992.)  

# Data analysis before recommendation
Due to large amount of data records of webpage views of users, the following problems will exist if Collaborative Filtering are applied directly without classification.
1) The model takes a lot of time to calculate the sparse matrix of users and items.
2) Different users focus on different information, and the recommendation results cannot meet the individual needs of users.

Therefore, my model make the classfication of webpages based on the webpage IP types before the recommendation and do the exploratory analysis. 

According to the results, 
1) **Users prefered to look for questions posted rather than asking questions or looking at long content to find what they need.**
2) **Most users came to the lawtime.cn to ask for help or consulting from lawyers.**

# Recommendation Model
You can find codes in the **WebRecom.py**. The procedure of model is showed in the picture below:



# Results
You can find the results in the **Resutl.csv** where "IP address", "URI_vieweds", "URI_recomm" and "Evaluation" are included.
