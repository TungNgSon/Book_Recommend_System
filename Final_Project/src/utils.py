"""
Helper functions cho recommendation system
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_book_info(isbn: str, books_df: pd.DataFrame) -> Dict:
    """Lấy thông tin chi tiết của sách"""
    book = books_df[books_df['ISBN'] == isbn]
    if book.empty:
        return None

    book = book.iloc[0]
    return {
        'isbn': isbn,
        'title': book['Book-Title'],
        'author': book['Book-Author'],
        'year': book['Year-Of-Publication'],
        'publisher': book['Publisher'],
        'image_s': book['Image-URL-S'],
        'image_m': book['Image-URL-M'],
        'image_l': book['Image-URL-L']
    }

def get_books_by_author(author: str, books_df: pd.DataFrame, exclude_isbn: str = None, limit: int = 5) -> List[str]:
    """Tìm sách cùng tác giả"""
    same_author = books_df[books_df['Book-Author'] == author]

    if exclude_isbn:
        same_author = same_author[same_author['ISBN'] != exclude_isbn]

    return same_author['ISBN'].head(limit).tolist()

def get_books_by_publisher(publisher: str, books_df: pd.DataFrame, exclude_isbn: str = None, limit: int = 5) -> List[str]:
    """Tìm sách cùng NXB"""
    same_publisher = books_df[books_df['Publisher'] == publisher]

    if exclude_isbn:
        same_publisher = same_publisher[same_publisher['ISBN'] != exclude_isbn]

    return same_publisher['ISBN'].head(limit).tolist()

def get_similar_books_cf(isbn: str, similarity_data, books_df: pd.DataFrame, limit: int = 10) -> List[Tuple[str, float]]:
    """
    Lấy sách tương tự dựa trên item-item collaborative filtering
    Returns: List of (isbn, similarity_score)

    Args:
        similarity_data: Có thể là DataFrame hoặc Dictionary
    """
    # Xử lý cả DataFrame và Dictionary
    if isinstance(similarity_data, dict):
        # Dictionary format: {isbn: [(similar_isbn, score), ...]}
        if isbn not in similarity_data:
            return []

        similar_items = similarity_data[isbn]

        # Filter và limit
        results = []
        for sim_isbn, score in similar_items[:limit]:
            if sim_isbn in books_df['ISBN'].values and score > 0.01:
                results.append((sim_isbn, score))

        return results

    else:
        # DataFrame format (legacy)
        if isbn not in similarity_data.columns:
            return []

        similar_scores = similarity_data[isbn].sort_values(ascending=False)
        similar_scores = similar_scores[similar_scores.index != isbn]
        top_similar = similar_scores.head(limit)

        results = []
        for sim_isbn, score in top_similar.items():
            if sim_isbn in books_df['ISBN'].values and score > 0.1:
                results.append((sim_isbn, score))

        return results

def get_svd_recommendations(user_id: int, model, books_df: pd.DataFrame,
                           ratings_df: pd.DataFrame, limit: int = 10,
                           exclude_isbns: List[str] = None) -> List[Tuple[str, float]]:
    """
    Lấy recommendations từ SVD model
    Returns: List of (isbn, predicted_rating)
    """
    # Lấy danh sách sách user chưa đọc
    user_rated = ratings_df[ratings_df['User-ID'] == user_id]['ISBN'].tolist()
    all_books = books_df['ISBN'].tolist()
    books_to_predict = [b for b in all_books if b not in user_rated]

    # Loại bỏ sách đã exclude
    if exclude_isbns:
        books_to_predict = [b for b in books_to_predict if b not in exclude_isbns]

    # Predict ratings
    predictions = []
    for isbn in books_to_predict:
        try:
            pred = model.predict(user_id, isbn)
            predictions.append((isbn, pred.est))
        except:
            continue

    # Sort theo predicted rating
    predictions.sort(key=lambda x: x[1], reverse=True)

    return predictions[:limit]


def compute_tfidf_similarity_realtime(clicked_isbn: str, target_isbn: str, books_df: pd.DataFrame) -> float:
    """
    Tính TF-IDF similarity giữa 2 sách dựa trên Title + Author (realtime)
    Chỉ dùng khi cần tính realtime, không có pre-computed matrix
    """
    clicked_book = books_df[books_df['ISBN'] == clicked_isbn]
    target_book = books_df[books_df['ISBN'] == target_isbn]
    
    if clicked_book.empty or target_book.empty:
        return 0.0
    
    # Kết hợp Title + Author
    clicked_text = f"{clicked_book.iloc[0]['Book-Title']} {clicked_book.iloc[0]['Book-Author']}"
    target_text = f"{target_book.iloc[0]['Book-Title']} {target_book.iloc[0]['Book-Author']}"
    
    # Tính TF-IDF
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([clicked_text, target_text])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
    except:
        return 0.0


def rerank_recommendations_with_clicks(
    recommendations: List[Tuple[str, float]], 
    clicked_books,  # List of strings hoặc list of tuples (isbn, timestamp)
    similarity_data,
    books_df: pd.DataFrame = None,
    boost_factor: float = 0.3,
    use_tfidf: bool = False,
    tfidf_boost_factor: float = 0.15,
    return_debug_info: bool = False,
    use_weighted_sum: bool = False,  # Nếu True: cộng dồn tất cả similarities, nếu False: chỉ lấy MAX
    use_recency_weight: bool = False,  # Nếu True: áp dụng recency weighting
    recency_decay_rate: float = 0.1  # Exponential decay rate (càng lớn = decay nhanh hơn)
) -> List[Tuple[str, float]]:
    """
    Rerank recommendations dựa trên sách user đã click
    Kết hợp Item-Item CF similarity và TF-IDF (Content-Based)
    
    Args:
        recommendations: List of (isbn, predicted_rating)
        clicked_books: List of ISBNs user đã click
        similarity_data: Item-Item CF similarity dictionary
        books_df: DataFrame chứa metadata sách (cần cho TF-IDF)
        boost_factor: Hệ số boost cho CF similarity (0.0 - 1.0)
        use_tfidf: Có sử dụng TF-IDF không
        tfidf_boost_factor: Hệ số boost cho TF-IDF similarity (0.0 - 1.0)
        return_debug_info: Có trả về debug info không
    
    Returns:
        Reranked list of (isbn, boosted_score)
        Nếu return_debug_info=True: (reranked_list, debug_info_dict)
    """
    if not clicked_books or not similarity_data:
        return recommendations if not return_debug_info else (recommendations, {})
    
    import time
    
    # Parse clicked_books: có thể là list of strings hoặc list of tuples
    clicked_with_timestamps = []
    current_time = time.time()
    
    for item in clicked_books:
        if isinstance(item, tuple):
            isbn, timestamp = item
            clicked_with_timestamps.append((isbn, timestamp))
        else:
            # Nếu là string, không có timestamp (treat as very recent)
            clicked_with_timestamps.append((item, current_time))
    
    # Sort theo timestamp (mới nhất trước) để tính recency weight
    clicked_with_timestamps.sort(key=lambda x: x[1], reverse=True)
    
    # Tính recency weights cho mỗi clicked book
    recency_weights = {}
    if use_recency_weight and clicked_with_timestamps:
        # Sách mới nhất có weight = 1.0
        # Sách cũ hơn có weight giảm dần theo exponential decay
        newest_time = clicked_with_timestamps[0][1]
        for isbn, timestamp in clicked_with_timestamps:
            time_diff = newest_time - timestamp  # Seconds
            # Exponential decay: weight = exp(-decay_rate * time_diff_in_minutes)
            time_diff_minutes = time_diff / 60.0
            weight = np.exp(-recency_decay_rate * time_diff_minutes)
            recency_weights[isbn] = weight
    else:
        # Không dùng recency weight, tất cả = 1.0
        for isbn, _ in clicked_with_timestamps:
            recency_weights[isbn] = 1.0
    
    # Tính boost score cho mỗi recommendation
    boosted_scores = {}
    debug_info = {}  # Lưu thông tin debug
    
    for isbn, base_score in recommendations:
        boosted_score = base_score
        
        # Thu thập tất cả CF similarities với tất cả clicked books
        cf_similarities = []  # List of (clicked_isbn, similarity_score, recency_weight)
        max_cf_similarity = 0.0
        best_cf_clicked = None
        
        for clicked_isbn, _ in clicked_with_timestamps:
            if clicked_isbn in similarity_data:
                similar_items = similarity_data[clicked_isbn]
                # Tìm similarity với isbn hiện tại
                for sim_isbn, sim_score in similar_items:
                    if sim_isbn == isbn:
                        recency_weight = recency_weights.get(clicked_isbn, 1.0)
                        weighted_sim = sim_score * recency_weight
                        cf_similarities.append((clicked_isbn, sim_score, recency_weight, weighted_sim))
                        
                        # Track max (cho debug)
                        if sim_score > max_cf_similarity:
                            max_cf_similarity = sim_score
                            best_cf_clicked = clicked_isbn
                        break
        
        # Thu thập tất cả TF-IDF similarities với tất cả clicked books
        tfidf_similarities = []  # List of (clicked_isbn, similarity_score, recency_weight)
        max_tfidf_similarity = 0.0
        best_tfidf_clicked = None
        
        if use_tfidf and books_df is not None:
            for clicked_isbn, _ in clicked_with_timestamps:
                tfidf_sim = compute_tfidf_similarity_realtime(clicked_isbn, isbn, books_df)
                if tfidf_sim > 0:
                    recency_weight = recency_weights.get(clicked_isbn, 1.0)
                    weighted_tfidf = tfidf_sim * recency_weight
                    tfidf_similarities.append((clicked_isbn, tfidf_sim, recency_weight, weighted_tfidf))
                    
                    # Track max (cho debug)
                    if tfidf_sim > max_tfidf_similarity:
                        max_tfidf_similarity = tfidf_sim
                        best_tfidf_clicked = clicked_isbn
        
        # Tính aggregated similarity (weighted sum hoặc max)
        if use_weighted_sum:
            # Strategy: Weighted sum với decay (ưu tiên similarity cao nhất)
            # Similarity cao nhất: weight = 1.0
            # Similarity thứ 2: weight = 0.5
            # Similarity thứ 3: weight = 0.25
            # ...
            
            # CF: Sort theo weighted similarity (đã có recency weight)
            if cf_similarities:
                sorted_cf = sorted(cf_similarities, key=lambda x: x[3], reverse=True)  # Sort by weighted_sim
                aggregated_cf = 0.0
                for idx, (clicked_isbn, raw_sim, recency_w, weighted_sim) in enumerate(sorted_cf):
                    position_weight = 1.0 / (2 ** idx)  # Decay factor theo vị trí
                    aggregated_cf += weighted_sim * position_weight
                # Normalize để không quá lớn (cap at 1.0)
                aggregated_cf = min(aggregated_cf, 1.0)
            else:
                aggregated_cf = 0.0
            
            # TF-IDF: Tương tự
            if use_tfidf and tfidf_similarities:
                sorted_tfidf = sorted(tfidf_similarities, key=lambda x: x[3], reverse=True)
                aggregated_tfidf = 0.0
                for idx, (clicked_isbn, raw_sim, recency_w, weighted_sim) in enumerate(sorted_tfidf):
                    position_weight = 1.0 / (2 ** idx)
                    aggregated_tfidf += weighted_sim * position_weight
                aggregated_tfidf = min(aggregated_tfidf, 1.0)
            else:
                aggregated_tfidf = 0.0
        else:
            # Strategy cũ: Chỉ lấy MAX (đã có recency weight trong weighted_sim)
            if cf_similarities:
                aggregated_cf = max([x[3] for x in cf_similarities])  # Max weighted similarity
            else:
                aggregated_cf = 0.0
            
            if use_tfidf and tfidf_similarities:
                aggregated_tfidf = max([x[3] for x in tfidf_similarities])
            else:
                aggregated_tfidf = 0.0
        
        # 3. Boost score từ CF similarity
        cf_boost = 0.0
        if aggregated_cf > 0:
            cf_boost = aggregated_cf * boost_factor * base_score
            boosted_score += cf_boost
        
        # 4. Boost score từ TF-IDF similarity
        tfidf_boost = 0.0
        if use_tfidf and aggregated_tfidf > 0:
            tfidf_boost = aggregated_tfidf * tfidf_boost_factor * base_score
            boosted_score += tfidf_boost
        
        boosted_scores[isbn] = boosted_score
        
        # Lưu debug info
        if return_debug_info:
            # Tạo dictionary để hiển thị chi tiết
            all_cf_details = {}
            for clicked_isbn, raw_sim, recency_w, weighted_sim in cf_similarities:
                all_cf_details[clicked_isbn] = {
                    'raw_similarity': raw_sim,
                    'recency_weight': recency_w,
                    'weighted_similarity': weighted_sim
                }
            
            all_tfidf_details = {}
            for clicked_isbn, raw_sim, recency_w, weighted_sim in tfidf_similarities:
                all_tfidf_details[clicked_isbn] = {
                    'raw_similarity': raw_sim,
                    'recency_weight': recency_w,
                    'weighted_similarity': weighted_sim
                }
            
            debug_info[isbn] = {
                'base_score': base_score,
                'cf_similarity': aggregated_cf if use_weighted_sum else max_cf_similarity,  # Aggregated hoặc max
                'cf_similarity_max': max_cf_similarity,  # Max raw similarity
                'cf_similarity_aggregated': aggregated_cf if use_weighted_sum else None,
                'cf_similarities_count': len(cf_similarities),
                'all_cf_similarities': all_cf_details,  # Chi tiết từng clicked book
                'tfidf_similarity': aggregated_tfidf if use_weighted_sum else max_tfidf_similarity,
                'tfidf_similarity_max': max_tfidf_similarity,
                'tfidf_similarity_aggregated': aggregated_tfidf if use_weighted_sum else None,
                'tfidf_similarities_count': len(tfidf_similarities),
                'all_tfidf_similarities': all_tfidf_details,
                'cf_boost': cf_boost,
                'tfidf_boost': tfidf_boost,
                'final_score': boosted_score,
                'best_cf_clicked': best_cf_clicked,
                'best_tfidf_clicked': best_tfidf_clicked,
                'use_weighted_sum': use_weighted_sum,
                'use_recency_weight': use_recency_weight
            }
    
    # Sort lại theo boosted score
    reranked = [(isbn, boosted_scores[isbn]) for isbn, _ in recommendations]
    reranked.sort(key=lambda x: x[1], reverse=True)
    
    if return_debug_info:
        return reranked, debug_info
    return reranked

def get_top_rated_books(ratings_df: pd.DataFrame, books_df: pd.DataFrame,
                       min_ratings: int = 10, limit: int = 10) -> List[Tuple[str, float, int]]:
    """
    Lấy sách có rating cao nhất
    Returns: List of (isbn, avg_rating, num_ratings)
    """
    # Tính avg rating và số lượng ratings
    book_stats = ratings_df.groupby('ISBN').agg({
        'Book-Rating': ['mean', 'count']
    }).reset_index()

    book_stats.columns = ['ISBN', 'avg_rating', 'num_ratings']

    # Filter sách có đủ ratings
    book_stats = book_stats[book_stats['num_ratings'] >= min_ratings]

    # Sort theo avg_rating
    book_stats = book_stats.sort_values('avg_rating', ascending=False)

    # Chỉ lấy sách có trong books_df
    book_stats = book_stats[book_stats['ISBN'].isin(books_df['ISBN'])]

    results = []
    for _, row in book_stats.head(limit).iterrows():
        results.append((row['ISBN'], row['avg_rating'], row['num_ratings']))

    return results

def explain_recommendation(isbn: str, book_info: Dict, reason: str, score: float = None) -> str:
    """Tạo explanation cho recommendation"""
    title = book_info['title']
    author = book_info['author']

    if reason == 'same_author':
        return f"📚 Cùng tác giả **{author}**"
    elif reason == 'same_publisher':
        return f"🏢 Cùng NXB **{book_info['publisher']}**"
    elif reason == 'collaborative':
        if score:
            return f"👥 {score*100:.1f}% người đọc sách bạn quan tâm cũng thích cuốn này"
        return "👥 Người đọc sách tương tự cũng thích"
    elif reason == 'personalized':
        if score:
            return f"⭐ Dự đoán bạn sẽ đánh giá {score:.1f}/10"
        return "⭐ Phù hợp với sở thích của bạn"
    elif reason == 'top_rated':
        if score:
            return f"🔥 Rating trung bình: {score:.1f}/10"
        return "🔥 Được đánh giá cao"
    else:
        return "💡 Gợi ý cho bạn"

def format_year(year):
    """Format năm xuất bản"""
    try:
        return int(year)
    except:
        return "N/A"

def get_random_books(books_df: pd.DataFrame, n: int = 20) -> List[str]:
    """Lấy random books để hiển thị trang chủ"""
    return books_df.sample(n=min(n, len(books_df)))['ISBN'].tolist()

def search_books(query: str, books_df: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    """
    Search sách theo tên hoặc tác giả
    """
    query = query.lower().strip()

    if not query:
        return pd.DataFrame()

    # Search trong title và author
    mask = (
        books_df['Book-Title'].str.lower().str.contains(query, na=False) |
        books_df['Book-Author'].str.lower().str.contains(query, na=False)
    )

    results = books_df[mask].head(limit)
    return results

def update_user_preference(clicked_isbn: str, books_df: pd.DataFrame, user_profile: Dict) -> Dict:
    """
    Update user preference dựa trên sách đã click
    Để sử dụng cho context-aware recommendation
    """
    book_info = get_book_info(clicked_isbn, books_df)

    if not book_info:
        return user_profile

    # Count tác giả
    author = book_info['author']
    if 'favorite_authors' not in user_profile:
        user_profile['favorite_authors'] = {}
    user_profile['favorite_authors'][author] = user_profile['favorite_authors'].get(author, 0) + 1

    # Count NXB
    publisher = book_info['publisher']
    if 'favorite_publishers' not in user_profile:
        user_profile['favorite_publishers'] = {}
    user_profile['favorite_publishers'][publisher] = user_profile['favorite_publishers'].get(publisher, 0) + 1

    # Count năm
    year = book_info['year']
    if 'favorite_years' not in user_profile:
        user_profile['favorite_years'] = {}
    user_profile['favorite_years'][year] = user_profile['favorite_years'].get(year, 0) + 1

    return user_profile

def is_cold_start_user(user_id: int, ratings_df: pd.DataFrame) -> bool:
    """
    Kiểm tra xem user có phải cold start không (chưa có rating trong training data)
    
    Args:
        user_id: User ID cần kiểm tra
        ratings_df: DataFrame chứa ratings
    
    Returns:
        True nếu user là cold start (chưa có rating), False nếu đã có rating
    """
    user_ratings = ratings_df[ratings_df['User-ID'] == user_id]
    return len(user_ratings) == 0

def get_fallback_recommendations(
    clicked_books,  # List of tuples (isbn, timestamp) hoặc list of strings
    similarity_data,
    books_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    limit: int = 10,
    exclude_isbns: List[str] = None
) -> List[Tuple[str, float]]:
    """
    Fallback recommendations khi SVD không hoạt động (cold start)
    
    Strategy:
    1. Nếu có clicked books → tìm similar books dựa trên CF + TF-IDF
    2. Nếu không có clicked books → trả về top-rated books
    
    Args:
        clicked_books: List of (isbn, timestamp) hoặc list of ISBNs
        similarity_data: Item-Item CF similarity dictionary
        books_df: DataFrame chứa metadata sách
        ratings_df: DataFrame chứa ratings
        limit: Số lượng recommendations cần trả về
        exclude_isbns: Danh sách ISBNs cần loại bỏ
    
    Returns:
        List of (isbn, score) - score có thể là similarity hoặc avg_rating
    """
    # Parse clicked_books để lấy ISBNs
    clicked_isbns = []
    if clicked_books:
        for item in clicked_books:
            if isinstance(item, tuple):
                clicked_isbns.append(item[0])
            else:
                clicked_isbns.append(item)
    
    # Loại bỏ duplicates
    clicked_isbns = list(set(clicked_isbns))
    
    # Strategy 1: Nếu có clicked books → tìm similar books
    if clicked_isbns:
        similar_books_dict = {}  # {isbn: max_similarity_score}
        
        for clicked_isbn in clicked_isbns:
            # Lấy similar books từ CF
            cf_similar = get_similar_books_cf(clicked_isbn, similarity_data, books_df, limit=10)
            for sim_isbn, cf_score in cf_similar:
                # Loại bỏ clicked books
                if sim_isbn in clicked_isbns:
                    continue
                # Loại bỏ exclude_isbns
                if exclude_isbns and sim_isbn in exclude_isbns:
                    continue
                # Lấy max similarity
                if sim_isbn not in similar_books_dict:
                    similar_books_dict[sim_isbn] = 0.0
                similar_books_dict[sim_isbn] = max(similar_books_dict[sim_isbn], cf_score)
            
            # Tính TF-IDF similarity với tất cả books (nếu cần)
            # Note: Có thể tính TF-IDF cho tất cả, nhưng sẽ chậm
            # Tạm thời chỉ dùng CF similarity
        
        # Convert dict thành list và sort
        similar_books = [(isbn, score) for isbn, score in similar_books_dict.items()]
        similar_books.sort(key=lambda x: x[1], reverse=True)
        
        # Nếu có đủ recommendations từ clicked books
        if len(similar_books) >= limit:
            return similar_books[:limit]
        
        # Nếu không đủ, bổ sung bằng top-rated books
        top_books = get_top_rated_books(ratings_df, books_df, min_ratings=20, limit=limit * 2)
        top_books_list = [(isbn, avg_rating) for isbn, avg_rating, _ in top_books]
        
        # Loại bỏ clicked books và exclude_isbns
        filtered_top = [
            (isbn, score) for isbn, score in top_books_list
            if isbn not in clicked_isbns and (not exclude_isbns or isbn not in exclude_isbns)
        ]
        
        # Combine: similar books + top-rated (loại bỏ duplicates)
        combined_isbns = set([isbn for isbn, _ in similar_books])
        for isbn, score in filtered_top:
            if isbn not in combined_isbns:
                similar_books.append((isbn, score))
                combined_isbns.add(isbn)
                if len(similar_books) >= limit:
                    break
        
        return similar_books[:limit]
    
    # Strategy 2: Không có clicked books → trả về top-rated books
    else:
        top_books = get_top_rated_books(ratings_df, books_df, min_ratings=20, limit=limit)
        results = [(isbn, avg_rating) for isbn, avg_rating, _ in top_books]
        
        # Loại bỏ exclude_isbns
        if exclude_isbns:
            results = [(isbn, score) for isbn, score in results if isbn not in exclude_isbns]
        
        return results