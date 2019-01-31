Name: Shwetha Konairy Suresha
ID: 1001556725

Language Used: Python 3.6

Code Structure:
1. Read all the documents in presidential debates folder
2. Fetch all the tokens and perform stemming
3. Calculate the tf for each token
4. Calculate the idf for each token
5. Calculate the weight of each token : $$w_{t,d} = (1+log_{10}{tf_{t,d}})\times(log_{10}{\frac{N}{df_t}})
6. Get the ranking of top 10 documents

