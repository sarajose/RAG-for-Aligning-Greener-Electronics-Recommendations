# Metrics

### Hit@k
Checks if at least one relevant document appears in the top-k retrieved documents for a query.
Example: If the gold standard says Document A is relevant and it appears in your top 5 results, Hit@5 = 1 for that query.
### Recall@k (not sure if relevant)
Measures the fraction of all relevant documents that appear in the top-k retrieved documents.
Example: If there are 3 relevant documents and 2 are in your top 10, Recall@10 = 2/3.
### Precision@k (not sure if relevant)
Measures the fraction of the top-k retrieved documents that are actually relevant.
Example: If 3 out of your top 5 retrieved documents are relevant, Precision@5 = 3/5.
### MRR (Mean Reciprocal Rank)
Looks at the rank position of the first relevant document in your results.
Example: If the first relevant document is at position 2, the reciprocal rank is 1/2. MRR is the average of this value across all queries.
### MAP (Mean Average Precision)
Computes the average precision for each query (precision at each rank where a relevant document is found), then averages across all queries.
Example: If relevant documents are at ranks 2 and 4, you compute precision at each of those ranks and average them.
### NDCG (Normalized Discounted Cumulative Gain)
Measures ranking quality, giving higher scores for relevant documents appearing higher in the list.
Example: If a relevant document is at rank 1, it contributes more to NDCG than if it’s at rank 10.