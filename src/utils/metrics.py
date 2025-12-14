"""
Evaluation Metrics for Recommendation Systems
Includes: Precision@K, Recall@K, NDCG, Diversity, Coverage
"""

import numpy as np
from collections import Counter
from sklearn.metrics import ndcg_score

class RecommendationMetrics:
    """
    Calculate various metrics for recommendation evaluation
    """
    
    @staticmethod
    def precision_at_k(predictions, ground_truth, k=10):
        """
        Precision@K: Proportion of relevant items in top-K
        
        Args:
            predictions: List of predicted item IDs (ordered by score)
            ground_truth: List of relevant item IDs
            k: Number of top items to consider
            
        Returns:
            float: Precision value [0, 1]
        """
        if k <= 0 or len(predictions) == 0:
            return 0.0
        
        top_k = predictions[:k]
        relevant = set(ground_truth)
        
        hits = len(set(top_k) & relevant)
        return hits / k
    
    @staticmethod
    def recall_at_k(predictions, ground_truth, k=10):
        """
        Recall@K: Proportion of relevant items retrieved in top-K
        
        Args:
            predictions: List of predicted item IDs
            ground_truth: List of relevant item IDs
            k: Number of top items to consider
            
        Returns:
            float: Recall value [0, 1]
        """
        if k <= 0 or len(ground_truth) == 0:
            return 0.0
        
        top_k = predictions[:k]
        relevant = set(ground_truth)
        
        hits = len(set(top_k) & relevant)
        return hits / len(relevant)
    
    @staticmethod
    def f1_at_k(predictions, ground_truth, k=10):
        """
        F1@K: Harmonic mean of Precision and Recall
        """
        precision = RecommendationMetrics.precision_at_k(predictions, ground_truth, k)
        recall = RecommendationMetrics.recall_at_k(predictions, ground_truth, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def ndcg_at_k(predictions, ground_truth_scores, k=10):
        """
        NDCG@K: Normalized Discounted Cumulative Gain
        Considers ranking order and relevance scores
        
        Args:
            predictions: List of predicted item IDs
            ground_truth_scores: Dict {item_id: relevance_score}
            k: Number of top items
            
        Returns:
            float: NDCG value [0, 1]
        """
        if k <= 0 or len(predictions) == 0:
            return 0.0
        
        # Get relevance scores for predictions
        predicted_relevance = [
            ground_truth_scores.get(item_id, 0.0) 
            for item_id in predictions[:k]
        ]
        
        # Get ideal ordering (sorted by relevance)
        ideal_relevance = sorted(
            ground_truth_scores.values(), 
            reverse=True
        )[:k]
        
        # Calculate NDCG
        try:
            ndcg = ndcg_score(
                [ideal_relevance], 
                [predicted_relevance]
            )
            return ndcg
        except:
            return 0.0
    
    @staticmethod
    def diversity_score(predictions, item_features, k=10):
        """
        Diversity: Average pairwise dissimilarity in top-K
        Higher = more diverse recommendations
        
        Args:
            predictions: List of predicted item IDs
            item_features: Dict {item_id: feature_vector} or category
            k: Number of top items
            
        Returns:
            float: Diversity score [0, 1]
        """
        top_k = predictions[:k]
        
        if len(top_k) < 2:
            return 0.0
        
        # If features are categories (strings)
        if isinstance(list(item_features.values())[0], str):
            unique_categories = len(set([
                item_features.get(item_id, '') 
                for item_id in top_k
            ]))
            return unique_categories / len(top_k)
        
        # If features are vectors
        else:
            from sklearn.metrics.pairwise import cosine_similarity
            
            vectors = [item_features[item_id] for item_id in top_k if item_id in item_features]
            
            if len(vectors) < 2:
                return 0.0
            
            # Calculate average pairwise dissimilarity
            similarities = cosine_similarity(vectors)
            
            # Get upper triangle (exclude diagonal)
            n = len(similarities)
            dissimilarities = []
            for i in range(n):
                for j in range(i+1, n):
                    dissimilarities.append(1 - similarities[i][j])
            
            return np.mean(dissimilarities) if dissimilarities else 0.0
    
    @staticmethod
    def coverage(all_predictions, total_items):
        """
        Coverage: Proportion of items that appear in any recommendation
        
        Args:
            all_predictions: List of lists (recommendations for multiple queries)
            total_items: Total number of items in catalog
            
        Returns:
            float: Coverage [0, 1]
        """
        if total_items == 0:
            return 0.0
        
        # Flatten all predictions
        recommended_items = set()
        for pred in all_predictions:
            recommended_items.update(pred)
        
        return len(recommended_items) / total_items
    
    @staticmethod
    def novelty(predictions, item_popularity, k=10):
        """
        Novelty: Average unexpectedness of recommendations
        Higher = recommends less popular (more novel) items
        
        Args:
            predictions: List of predicted item IDs
            item_popularity: Dict {item_id: popularity_score} (higher = more popular)
            k: Number of top items
            
        Returns:
            float: Novelty score [0, 1]
        """
        top_k = predictions[:k]
        
        if len(top_k) == 0:
            return 0.0
        
        # Normalize popularity to [0, 1]
        max_pop = max(item_popularity.values()) if item_popularity else 1
        
        novelty_scores = []
        for item_id in top_k:
            pop = item_popularity.get(item_id, 0)
            # Novelty = 1 - normalized_popularity
            novelty = 1 - (pop / max_pop) if max_pop > 0 else 1.0
            novelty_scores.append(novelty)
        
        return np.mean(novelty_scores)
    
    @staticmethod
    def serendipity(predictions, ground_truth, item_similarity, k=10, threshold=0.7):
        """
        Serendipity: Relevant but unexpected recommendations
        
        Args:
            predictions: List of predicted item IDs
            ground_truth: List of relevant items (user's history)
            item_similarity: Dict {(item1, item2): similarity_score}
            k: Number of top items
            threshold: Similarity threshold (below = unexpected)
            
        Returns:
            float: Serendipity score [0, 1]
        """
        top_k = predictions[:k]
        
        if len(top_k) == 0 or len(ground_truth) == 0:
            return 0.0
        
        serendipitous_items = 0
        
        for pred_item in top_k:
            # Check if relevant (in some extended ground truth)
            is_relevant = pred_item in ground_truth  # Simplified
            
            # Check if unexpected (low similarity to history)
            is_unexpected = True
            for hist_item in ground_truth:
                sim = item_similarity.get((pred_item, hist_item), 
                                         item_similarity.get((hist_item, pred_item), 0))
                if sim > threshold:
                    is_unexpected = False
                    break
            
            if is_relevant and is_unexpected:
                serendipitous_items += 1
        
        return serendipitous_items / k


class MetricsCalculator:
    """
    Helper class to calculate all metrics at once
    """
    
    def __init__(self):
        self.metrics = RecommendationMetrics()
    
    def calculate_all(self, predictions, ground_truth, item_data, k=10):
        """
        Calculate all metrics for a recommendation result
        
        Args:
            predictions: List of predicted item IDs
            ground_truth: Dict with keys: 'relevant_items', 'scores', 'popularity'
            item_data: Dict with item features/categories
            k: Top-K
            
        Returns:
            Dict of all metrics
        """
        results = {}
        
        # Accuracy metrics
        results['precision@k'] = self.metrics.precision_at_k(
            predictions, ground_truth.get('relevant_items', []), k
        )
        results['recall@k'] = self.metrics.recall_at_k(
            predictions, ground_truth.get('relevant_items', []), k
        )
        results['f1@k'] = self.metrics.f1_at_k(
            predictions, ground_truth.get('relevant_items', []), k
        )
        
        # Ranking quality
        if 'scores' in ground_truth:
            results['ndcg@k'] = self.metrics.ndcg_at_k(
                predictions, ground_truth['scores'], k
            )
        
        # Diversity & Discovery
        results['diversity'] = self.metrics.diversity_score(
            predictions, item_data, k
        )
        
        if 'popularity' in ground_truth:
            results['novelty'] = self.metrics.novelty(
                predictions, ground_truth['popularity'], k
            )
        
        return results


if __name__ == "__main__":
    # Test metrics
    print("[TEST] Evaluation Metrics")
    
    # Sample data
    predictions = ['book1', 'book2', 'book3', 'book4', 'book5']
    ground_truth = ['book1', 'book3', 'book6']
    
    metrics = RecommendationMetrics()
    
    print(f"Predictions: {predictions}")
    print(f"Ground truth: {ground_truth}")
    print()
    
    precision = metrics.precision_at_k(predictions, ground_truth, k=5)
    recall = metrics.recall_at_k(predictions, ground_truth, k=5)
    f1 = metrics.f1_at_k(predictions, ground_truth, k=5)
    
    print(f"Precision@5: {precision:.3f}")
    print(f"Recall@5: {recall:.3f}")
    print(f"F1@5: {f1:.3f}")
    
    # Diversity test
    item_categories = {
        'book1': 'Fiction',
        'book2': 'Fiction',
        'book3': 'History',
        'book4': 'Science',
        'book5': 'Biography'
    }
    
    diversity = metrics.diversity_score(predictions, item_categories, k=5)
    print(f"Diversity: {diversity:.3f} (4 unique categories / 5 books)")