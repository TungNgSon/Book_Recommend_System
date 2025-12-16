"""
Book Recommendation System - Streamlit App
Hệ thống gợi ý sách sử dụng SVD + Item-Item CF + Content-Based
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Import utils
from utils import (
    get_book_info, get_books_by_author, get_books_by_publisher,
    get_similar_books_cf, get_svd_recommendations, get_top_rated_books,
    explain_recommendation, get_random_books, search_books,
    update_user_preference, rerank_recommendations_with_clicks,
    is_cold_start_user, get_fallback_recommendations
)

# Page config
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
<style>
    .book-card {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
        background: white;
    }
    .book-title {
        font-weight: bold;
        font-size: 1.1rem;
        color: #1f77b4;
    }
    .book-meta {
        color: #666;
        font-size: 0.9rem;
    }
    .reason-tag {
        background: #e8f4f8;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-size: 0.85rem;
        display: inline-block;
        margin-top: 0.5rem;
    }
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1.5rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def get_data_path(relative_path: str) -> Path:
    """
    Tìm đường dẫn data/model file, hỗ trợ cả local và deployment
    Thử các paths:
    1. Relative từ src/ (../dataset/...)
    2. Absolute từ project root (./dataset/...)
    3. Từ current working directory
    """
    # Path từ src/ lên parent (local development)
    path1 = Path(__file__).parent.parent / relative_path
    
    # Path từ current working directory (deployment)
    path2 = Path(relative_path)
    
    # Path từ project root (nếu chạy từ root)
    path3 = Path('.') / relative_path
    
    # Thử từng path
    for path in [path1, path2, path3]:
        if path.exists():
            return path
    
    # Nếu không tìm thấy, trả về path đầu tiên (để hiển thị lỗi)
    return path1


@st.cache_resource
def load_data():
    """Load tất cả data và models"""
    try:
        # Load books
        books_path = get_data_path('dataset/cleaned/Books_cleaned.csv')
        books = pd.read_csv(books_path)

        # Load users
        users_path = get_data_path('dataset/cleaned/Users_cleaned.csv')
        users = pd.read_csv(users_path)

        # Load ratings
        ratings_path = get_data_path('dataset/cleaned/Ratings_cleaned.csv')
        ratings = pd.read_csv(ratings_path)

        # Load SVD model
        model_path = get_data_path('notebook/saved_models/svd_model.pkl')
        with open(model_path, 'rb') as f:
            svd_model = pickle.load(f)

        # Load item similarity matrix
        similarity_path = get_data_path('dataset/cleaned/item_similarity.pkl')
        with open(similarity_path, 'rb') as f:
            item_similarity = pickle.load(f)

        return books, users, ratings, svd_model, item_similarity

    except FileNotFoundError as e:
        st.error(f"❌ Không tìm thấy file: {e}")
        st.info("💡 Hãy chạy `python compute_similarity.py` trước để tạo item_similarity.pkl")
        st.info("💡 Đảm bảo các file data và models đã được copy vào đúng vị trí")
        st.stop()
    except Exception as e:
        st.error(f"❌ Lỗi khi load data: {e}")
        st.stop()


def display_book_card(isbn, books_df, reason="", score=None, show_button=True, key_suffix=""):
    """Hiển thị thông tin sách dạng card"""
    book_info = get_book_info(isbn, books_df)

    if not book_info:
        return

    col1, col2 = st.columns([1, 3])

    with col1:
        # Hiển thị ảnh
        try:
            st.image(book_info['image_m'], use_container_width=True)
        except:
            st.image("https://via.placeholder.com/150x200?text=No+Image", use_container_width=True)

    with col2:
        st.markdown(f"<div class='book-title'>{book_info['title']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='book-meta'>✍️ {book_info['author']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='book-meta'>📅 {book_info['year']} | 🏢 {book_info['publisher']}</div>",
                    unsafe_allow_html=True)

        if reason:
            explanation = explain_recommendation(isbn, book_info, reason, score)
            st.markdown(f"<div class='reason-tag'>{explanation}</div>", unsafe_allow_html=True)

        if show_button:
            # Tạo key duy nhất bằng cách kết hợp isbn, reason và key_suffix
            unique_key = f"btn_{reason}_{isbn}_{key_suffix}" if key_suffix else f"btn_{reason}_{isbn}"
            if st.button(f"👁️ Xem chi tiết", key=unique_key):
                st.session_state.selected_book = isbn
                st.rerun()


def show_book_detail(isbn, books_df, ratings_df, svd_model, similarity_df):
    """Hiển thị chi tiết sách và các gợi ý liên quan"""
    book_info = get_book_info(isbn, books_df)

    if not book_info:
        st.error("❌ Không tìm thấy thông tin sách")
        return

    # Lưu vào history với timestamp (để tính recency weight)
    import time
    if 'clicked_books' not in st.session_state:
        st.session_state.clicked_books = []  # List of tuples: [(isbn, timestamp), ...]

    # Kiểm tra xem isbn đã có chưa (chỉ lấy isbn, bỏ qua timestamp)
    existing_isbns = [item[0] if isinstance(item, tuple) else item for item in st.session_state.clicked_books]
    
    if isbn not in existing_isbns:
        # Thêm mới với timestamp hiện tại
        st.session_state.clicked_books.append((isbn, time.time()))
    else:
        # Nếu đã có, cập nhật timestamp (move to recent) - xóa cũ và thêm mới
        st.session_state.clicked_books = [
            item for item in st.session_state.clicked_books
            if (item[0] if isinstance(item, tuple) else item) != isbn
        ]
        st.session_state.clicked_books.append((isbn, time.time()))
        # Update user preference
        if 'user_profile' not in st.session_state:
            st.session_state.user_profile = {}
        st.session_state.user_profile = update_user_preference(isbn, books_df, st.session_state.user_profile)

    # Nút quay lại
    if st.button("⬅️ Quay lại"):
        st.session_state.selected_book = None
        st.rerun()

    st.markdown("---")

    # Thông tin chi tiết sách
    col1, col2 = st.columns([1, 2])

    with col1:
        try:
            st.image(book_info['image_l'], use_container_width=True)
        except:
            st.image("https://via.placeholder.com/300x400?text=No+Image", use_container_width=True)

    with col2:
        st.markdown(f"# {book_info['title']}")
        st.markdown(f"### ✍️ {book_info['author']}")
        st.markdown(f"**📅 Năm xuất bản:** {book_info['year']}")
        st.markdown(f"**🏢 Nhà xuất bản:** {book_info['publisher']}")
        st.markdown(f"**📖 ISBN:** {book_info['isbn']}")

        # Thống kê ratings
        book_ratings = ratings_df[ratings_df['ISBN'] == isbn]
        if not book_ratings.empty:
            avg_rating = book_ratings['Book-Rating'].mean()
            num_ratings = len(book_ratings)
            st.markdown(f"**⭐ Rating:** {avg_rating:.1f}/10 ({num_ratings} đánh giá)")

    st.markdown("---")

    # SECTION 1: Người đọc sách này cũng thích (Item-Item CF)
    st.markdown("<div class='section-header'><h3>👥 Người đọc sách này cũng thích</h3></div>", unsafe_allow_html=True)

    similar_cf = get_similar_books_cf(isbn, similarity_df, books_df, limit=5)

    if similar_cf:
        cols = st.columns(5)
        for idx, (sim_isbn, score) in enumerate(similar_cf):
            with cols[idx]:
                display_book_card(sim_isbn, books_df, reason='collaborative', score=score, show_button=True, key_suffix=f"cf_{idx}")
    else:
        st.info("Chưa có dữ liệu collaborative filtering cho sách này")

    # SECTION 2: Sách cùng tác giả
    st.markdown("<div class='section-header'><h3>📚 Sách khác của tác giả</h3></div>", unsafe_allow_html=True)

    same_author = get_books_by_author(book_info['author'], books_df, exclude_isbn=isbn, limit=5)

    if same_author:
        cols = st.columns(5)
        for idx, auth_isbn in enumerate(same_author):
            with cols[idx]:
                display_book_card(auth_isbn, books_df, reason='same_author', show_button=True, key_suffix=f"auth_{idx}")
    else:
        st.info("Không tìm thấy sách khác của tác giả này")

    # SECTION 3: Sách cùng NXB
    st.markdown("<div class='section-header'><h3>🏢 Sách cùng nhà xuất bản</h3></div>", unsafe_allow_html=True)

    same_publisher = get_books_by_publisher(book_info['publisher'], books_df, exclude_isbn=isbn, limit=5)

    if same_publisher:
        cols = st.columns(5)
        for idx, pub_isbn in enumerate(same_publisher):
            with cols[idx]:
                display_book_card(pub_isbn, books_df, reason='same_publisher', show_button=True, key_suffix=f"pub_{idx}")
    else:
        st.info("Không tìm thấy sách khác của NXB này")

    # SECTION 4: Gợi ý riêng cho bạn (SVD)
    if 'selected_user' in st.session_state and st.session_state.selected_user:
        st.markdown("<div class='section-header'><h3>⭐ Gợi ý riêng cho bạn</h3></div>", unsafe_allow_html=True)

        user_id = st.session_state.selected_user
        exclude_list = st.session_state.clicked_books if 'clicked_books' in st.session_state else [isbn]

        personalized = get_svd_recommendations(
            user_id, svd_model, books_df, ratings_df,
            limit=5, exclude_isbns=exclude_list
        )

        if personalized:
            cols = st.columns(5)
            for idx, (rec_isbn, pred_rating) in enumerate(personalized):
                with cols[idx]:
                    display_book_card(rec_isbn, books_df, reason='personalized', score=pred_rating, show_button=True, key_suffix=f"detail_{idx}")


def show_data_analysis(books_df, users_df, ratings_df):
    """Hiển thị phân tích và trực quan hóa dữ liệu"""
    st.title("📊 Phân tích & Trực quan hóa Dữ liệu")
    
    # Set style cho plots
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10
    
    # 1. Phân bố Rating (Histogram)
    st.markdown("### 📈 1. Phân bố Rating")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ratings_df['Book-Rating'].hist(bins=20, ax=ax1, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Rating', fontsize=12)
    ax1.set_ylabel('Số lượng đánh giá', fontsize=12)
    ax1.set_title('Phân bố Rating của người dùng', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)
    plt.close(fig1)
    
    # Thống kê rating
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rating trung bình", f"{ratings_df['Book-Rating'].mean():.2f}")
    with col2:
        st.metric("Rating trung vị", f"{ratings_df['Book-Rating'].median():.2f}")
    with col3:
        st.metric("Rating cao nhất", f"{ratings_df['Book-Rating'].max()}")
    with col4:
        st.metric("Rating thấp nhất", f"{ratings_df['Book-Rating'].min()}")
    
    st.markdown("---")
    
    # 2. Top Items (Bar Chart)
    st.markdown("### 🔥 2. Top 20 Sách được đánh giá cao nhất")
    
    # Tính toán top books
    book_ratings = ratings_df.groupby('ISBN').agg({
        'Book-Rating': ['mean', 'count']
    }).reset_index()
    book_ratings.columns = ['ISBN', 'avg_rating', 'num_ratings']
    book_ratings = book_ratings[book_ratings['num_ratings'] >= 20]  # Ít nhất 20 đánh giá
    book_ratings = book_ratings.sort_values('avg_rating', ascending=False).head(20)
    
    # Merge với books_df để lấy title
    top_books_merged = book_ratings.merge(books_df[['ISBN', 'Book-Title']], on='ISBN', how='left')
    top_books_merged['Book-Title'] = top_books_merged['Book-Title'].apply(
        lambda x: x[:50] + '...' if len(str(x)) > 50 else x
    )
    
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    bars = ax2.barh(range(len(top_books_merged)), top_books_merged['avg_rating'], color='coral')
    ax2.set_yticks(range(len(top_books_merged)))
    ax2.set_yticklabels(top_books_merged['Book-Title'], fontsize=9)
    ax2.set_xlabel('Rating trung bình', fontsize=12)
    ax2.set_title('Top 20 Sách được đánh giá cao nhất (≥20 đánh giá)', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Thêm giá trị rating trên mỗi bar
    for i, (idx, row) in enumerate(top_books_merged.iterrows()):
        ax2.text(row['avg_rating'] + 0.1, i, f"{row['avg_rating']:.2f} ({int(row['num_ratings'])} đánh giá)", 
                va='center', fontsize=8)
    
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)
    
    st.markdown("---")
    
    # 3. Tần suất nhóm sản phẩm - Top Authors (Bar Chart)
    st.markdown("### ✍️ 3. Top 15 Tác giả có nhiều sách nhất")
    
    author_counts = books_df['Book-Author'].value_counts().head(15)
    
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    bars = ax3.barh(range(len(author_counts)), author_counts.values, color='lightgreen')
    ax3.set_yticks(range(len(author_counts)))
    ax3.set_yticklabels(author_counts.index, fontsize=10)
    ax3.set_xlabel('Số lượng sách', fontsize=12)
    ax3.set_title('Top 15 Tác giả có nhiều sách nhất', fontsize=14, fontweight='bold')
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Thêm giá trị trên mỗi bar
    for i, (author, count) in enumerate(author_counts.items()):
        ax3.text(count + 5, i, str(count), va='center', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)
    
    st.markdown("---")
    
    # 4. Tần suất nhóm sản phẩm - Top Publishers (Bar Chart)
    st.markdown("### 🏢 4. Top 15 Nhà xuất bản có nhiều sách nhất")
    
    publisher_counts = books_df['Publisher'].value_counts().head(15)
    
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    bars = ax4.barh(range(len(publisher_counts)), publisher_counts.values, color='plum')
    ax4.set_yticks(range(len(publisher_counts)))
    ax4.set_yticklabels(publisher_counts.index, fontsize=10)
    ax4.set_xlabel('Số lượng sách', fontsize=12)
    ax4.set_title('Top 15 Nhà xuất bản có nhiều sách nhất', fontsize=14, fontweight='bold')
    ax4.invert_yaxis()
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Thêm giá trị trên mỗi bar
    for i, (publisher, count) in enumerate(publisher_counts.items()):
        ax4.text(count + 5, i, str(count), va='center', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)
    
    st.markdown("---")
    
    # 5. Heatmap - Rating theo năm xuất bản và số lượng đánh giá
    st.markdown("### 🔥 5. Heatmap: Rating trung bình theo Năm xuất bản")
    
    # Merge ratings với books để lấy năm xuất bản
    ratings_with_year = ratings_df.merge(
        books_df[['ISBN', 'Year-Of-Publication']], 
        on='ISBN', 
        how='left'
    )
    
    # Lọc năm hợp lệ (1900-2024)
    ratings_with_year = ratings_with_year[
        (ratings_with_year['Year-Of-Publication'] >= 1900) & 
        (ratings_with_year['Year-Of-Publication'] <= 2024)
    ]
    
    # Tính rating trung bình theo năm
    year_rating = ratings_with_year.groupby('Year-Of-Publication')['Book-Rating'].mean().reset_index()
    year_rating.columns = ['Year', 'Avg_Rating']
    
    # Tạo pivot table cho heatmap (nhóm theo thập kỷ)
    ratings_with_year['Decade'] = (ratings_with_year['Year-Of-Publication'] // 10) * 10
    decade_rating = ratings_with_year.groupby('Decade')['Book-Rating'].mean().reset_index()
    
    # Tạo heatmap data
    heatmap_data = decade_rating.set_index('Decade')['Book-Rating'].to_frame().T
    
    fig5, ax5 = plt.subplots(figsize=(14, 4))
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd', 
                cbar_kws={'label': 'Rating trung bình'}, ax=ax5, linewidths=0.5)
    ax5.set_title('Rating trung bình theo Thập kỷ xuất bản', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Thập kỷ', fontsize=12)
    ax5.set_ylabel('')
    ax5.set_yticklabels(['Rating TB'], rotation=0)
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close(fig5)
    
    # Hiển thị bảng chi tiết
    with st.expander("📋 Xem chi tiết Rating theo năm"):
        st.dataframe(year_rating.sort_values('Year', ascending=False), use_container_width=True)
    
    st.markdown("---")
    
    # 6. Thống kê tổng quan
    st.markdown("### 📊 6. Thống kê Tổng quan")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📚 Sách")
        st.metric("Tổng số sách", f"{len(books_df):,}")
        st.metric("Số tác giả", f"{books_df['Book-Author'].nunique():,}")
        st.metric("Số nhà xuất bản", f"{books_df['Publisher'].nunique():,}")
    
    with col2:
        st.markdown("#### 👥 Người dùng")
        st.metric("Tổng số users", f"{len(users_df):,}")
        st.metric("Số location", f"{users_df['Location'].nunique():,}")
        avg_age = users_df['Age'].mean()
        st.metric("Tuổi trung bình", f"{avg_age:.1f}")
    
    with col3:
        st.markdown("#### ⭐ Đánh giá")
        st.metric("Tổng số ratings", f"{len(ratings_df):,}")
        st.metric("Sách được đánh giá", f"{ratings_df['ISBN'].nunique():,}")
        st.metric("Users đã đánh giá", f"{ratings_df['User-ID'].nunique():,}")


def main():
    """Main app"""

    # Load data
    with st.spinner("🔄 Đang load dữ liệu..."):
        books_df, users_df, ratings_df, svd_model, similarity_df = load_data()

    # Initialize session state
    if 'selected_book' not in st.session_state:
        st.session_state.selected_book = None
    if 'clicked_books' not in st.session_state:
        st.session_state.clicked_books = []
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {}
    if 'last_user_id' not in st.session_state:
        st.session_state.last_user_id = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Gợi ý sách"
    
    # Sidebar (chung cho cả 2 tabs)
    with st.sidebar:
        st.title("📚 Book Recommender")

        # User selection - Cải thiện với text input
        st.subheader("👤 Chọn User ID")
        
        # Cache user IDs list
        if 'user_ids_list' not in st.session_state:
            st.session_state.user_ids_list = sorted(users_df['User-ID'].unique())
        
        # Text input với placeholder
        user_input = st.text_input(
            "Nhập User ID (hoặc để trống)",
            value="",
            placeholder="VD: 276847",
            key="user_id_input"
        )
        
        # Validate và convert user ID
        selected_user = None
        if user_input.strip():
            try:
                user_id = int(user_input.strip())
                if user_id in st.session_state.user_ids_list:
                    selected_user = user_id
                else:
                    st.warning(f"⚠️ User ID {user_id} không tồn tại trong hệ thống")
                    # Gợi ý user ID gần nhất
                    user_ids_array = np.array(st.session_state.user_ids_list)
                    closest_idx = np.abs(user_ids_array - user_id).argmin()
                    closest_user = user_ids_array[closest_idx]
                    st.info(f"💡 Gợi ý: User ID {closest_user} (gần nhất)")
            except ValueError:
                st.error("❌ Vui lòng nhập số User ID hợp lệ")
        
        # Reset clicked_books và user_profile khi user thay đổi
        # CHỈ reset khi: đổi từ user này sang user khác (KHÔNG reset khi None → user ID)
        # Mục đích: Giữ clicked books từ anonymous để xử lý cold start
        if st.session_state.last_user_id is not None and st.session_state.last_user_id != selected_user:
            st.session_state.clicked_books = []
            st.session_state.user_profile = {}
            st.session_state.selected_book = None  # Reset selected book khi đổi user
            # Xóa cache random books để refresh
            if 'random_books_list' in st.session_state:
                del st.session_state.random_books_list
        st.session_state.selected_user = selected_user
        st.session_state.last_user_id = selected_user

        # Hiển thị User ID hiện tại (nếu có)
        if selected_user:
            user_info = users_df[users_df['User-ID'] == selected_user].iloc[0]
            st.success(f"✅ User ID: **{selected_user}**")
            st.info(f"📍 {user_info['Location']}\n\n🎂 Tuổi: {user_info['Age']}")
            
            # Nút xóa User
            if st.button("🗑️ Xóa User", use_container_width=True):
                st.session_state.selected_user = None
                st.rerun()
        else:
            st.info("💡 Nhập User ID để xem gợi ý cá nhân hóa")

        st.markdown("---")

        # Search - Cải thiện với container scrollable
        st.subheader("🔍 Tìm kiếm sách")
        search_query = st.text_input(
            "Nhập tên sách hoặc tác giả",
            placeholder="VD: Harry Potter, J.K. Rowling...",
            key="search_input"
        )

        if search_query and len(search_query.strip()) >= 2:
            with st.spinner("🔍 Đang tìm kiếm..."):
                search_results = search_books(search_query, books_df, limit=20)
            
            if not search_results.empty:
                st.success(f"✅ Tìm thấy {len(search_results)} kết quả")

                # Container scrollable cho kết quả
                with st.container():
                    for idx, (_, book) in enumerate(search_results.iterrows()):
                        # Hiển thị compact hơn
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            title = book['Book-Title'][:60] + "..." if len(book['Book-Title']) > 60 else book['Book-Title']
                            author = book['Book-Author'][:30] + "..." if len(book['Book-Author']) > 30 else book['Book-Author']
                            st.markdown(f"**{title}**")
                            st.caption(f"✍️ {author}")
                        
                        with col2:
                            if st.button("👉", key=f"search_btn_{book['ISBN']}", help="Xem chi tiết"):
                                st.session_state.selected_book = book['ISBN']
                                st.rerun()
                        
                        if idx < len(search_results) - 1:
                            st.divider()
            else:
                st.info("🔍 Không tìm thấy kết quả. Thử từ khóa khác!")
        elif search_query and len(search_query.strip()) < 2:
            st.info("💡 Nhập ít nhất 2 ký tự để tìm kiếm")

        st.markdown("---")

        # History
        if st.session_state.clicked_books:
            st.subheader("📜 Lịch sử xem")
            st.write(f"Đã xem {len(st.session_state.clicked_books)} sách")

            if st.button("🗑️ Xóa lịch sử"):
                st.session_state.clicked_books = []
                st.session_state.user_profile = {}
                st.rerun()

        # Stats
        st.markdown("---")
        st.subheader("📊 Thống kê")
        st.metric("Tổng số sách", f"{len(books_df):,}")
        st.metric("Tổng số users", f"{len(users_df):,}")
        st.metric("Tổng số ratings", f"{len(ratings_df):,}")
    
    # Navigation tabs
    tab1, tab2 = st.tabs(["📚 Gợi ý sách", "📊 Phân tích dữ liệu"])
    
    with tab1:
        # Main content
        st.title("📚 Hệ Thống Gợi Ý Sách")

        # Hiển thị book detail nếu đã chọn
        if st.session_state.selected_book:
            show_book_detail(
                st.session_state.selected_book,
                books_df, ratings_df, svd_model, similarity_df
            )
        else:
            # Trang chủ
            st.markdown("### Chào mừng đến với hệ thống gợi ý sách!")
            st.markdown("Chọn user ID ở sidebar và khám phá các gợi ý sách phù hợp với bạn.")

        # Top rated books
        st.markdown("<div class='section-header'><h3>🔥 Sách được đánh giá cao nhất</h3></div>", unsafe_allow_html=True)

        top_books = get_top_rated_books(ratings_df, books_df, min_ratings=20, limit=10)

        cols = st.columns(5)
        for idx, (isbn, avg_rating, num_ratings) in enumerate(top_books[:5]):
            with cols[idx]:
                display_book_card(isbn, books_df, reason='top_rated', score=avg_rating, show_button=True, key_suffix=f"top_{idx}")

        cols = st.columns(5)
        for idx, (isbn, avg_rating, num_ratings) in enumerate(top_books[5:10]):
            with cols[idx]:
                display_book_card(isbn, books_df, reason='top_rated', score=avg_rating, show_button=True, key_suffix=f"top_{idx+5}")

        # Random books để explore
        st.markdown("<div class='section-header'><h3>🎲 Khám phá ngẫu nhiên</h3></div>", unsafe_allow_html=True)

        # Cache random books trong session_state để tránh thay đổi khi rerun
        if 'random_books_list' not in st.session_state:
            st.session_state.random_books_list = get_random_books(books_df, n=10)
        random_isbns = st.session_state.random_books_list

        cols = st.columns(5)
        for idx, isbn in enumerate(random_isbns[:5]):
            with cols[idx]:
                display_book_card(isbn, books_df, reason='random', show_button=True, key_suffix=f"rand_{idx}")

        cols = st.columns(5)
        for idx, isbn in enumerate(random_isbns[5:10]):
            with cols[idx]:
                display_book_card(isbn, books_df, reason='random', show_button=True, key_suffix=f"rand_{idx+5}")

        # Personalized nếu đã chọn user
        if st.session_state.selected_user:
            st.markdown("<div class='section-header'><h3>⭐ Gợi ý dành riêng cho bạn</h3></div>",
                        unsafe_allow_html=True)

            user_id = st.session_state.selected_user
            
            # Kiểm tra cold start user
            is_new_user = is_cold_start_user(user_id, ratings_df)
            
            # Lấy clicked_books (có thể từ anonymous hoặc sau khi login)
            clicked_books_data = st.session_state.clicked_books.copy() if st.session_state.clicked_books else []
            # Extract ISBNs để exclude (có thể là list of tuples hoặc list of strings)
            exclude_list = [
                item[0] if isinstance(item, tuple) else item 
                for item in clicked_books_data
            ]

            # Try SVD first
            personalized = get_svd_recommendations(
                user_id, svd_model, books_df, ratings_df,
                limit=15,  # Lấy nhiều hơn để rerank
                exclude_isbns=exclude_list if exclude_list else None
            )

            # Xử lý cold start: Nếu SVD fail hoặc trả về ít (user mới)
            use_fallback = False
            if not personalized or (is_new_user and len(personalized) < 5):
                use_fallback = True
                personalized = get_fallback_recommendations(
                    clicked_books_data,
                    similarity_df,
                    books_df,
                    ratings_df,
                    limit=15,
                    exclude_isbns=exclude_list if exclude_list else None
                )
                
                # User feedback cho cold start
                if is_new_user:
                    if clicked_books_data:
                        st.info("💡 Bạn là user mới! Chúng tôi đang gợi ý dựa trên sách bạn đã xem khi chưa đăng nhập.")
                    else:
                        st.info("💡 Bạn là user mới! Hãy khám phá sách phổ biến hoặc click vào sách để nhận gợi ý cá nhân hóa.")
                elif not personalized:
                    st.warning("⚠️ Không thể tạo gợi ý từ SVD. Đang sử dụng phương pháp thay thế.")

            # Rerank dựa trên clicked books (nếu có và không phải fallback hoặc fallback nhưng có clicked books)
            debug_info = None
            if personalized and exclude_list and (not use_fallback or clicked_books_data):
                personalized, debug_info = rerank_recommendations_with_clicks(
                    personalized,
                    clicked_books_data,  # Truyền list of tuples (isbn, timestamp) để tính recency
                    similarity_df,
                    books_df=books_df,  # Cần cho TF-IDF
                    boost_factor=0.25,  # Boost 25% cho CF similarity
                    use_tfidf=True,     # Bật TF-IDF
                    tfidf_boost_factor=0.15,  # Boost 15% cho TF-IDF similarity
                    return_debug_info=True,  # Trả về debug info
                    use_weighted_sum=True,  # Bật weighted sum (xem xét tất cả clicked books)
                    use_recency_weight=True,  # Bật recency weighting (ưu tiên sách gần đây)
                    recency_decay_rate=0.1  # Exponential decay rate (0.1 = decay chậm)
                )
                # Chỉ hiển thị caption nếu không phải fallback (vì fallback đã có message riêng)
                if not use_fallback:
                    st.caption("🔄 Gợi ý đã được cập nhật dựa trên sách bạn đã xem (CF + Content-Based)")
                
                # Debug panel - Hiển thị chi tiết rerank
                with st.expander("🔍 Xem chi tiết rerank (Debug)", expanded=False):
                    st.markdown("### 📚 Sách bạn đã xem:")
                    for clicked_isbn in exclude_list:
                        clicked_book = books_df[books_df['ISBN'] == clicked_isbn]
                        if not clicked_book.empty:
                            st.write(f"- **{clicked_book.iloc[0]['Book-Title']}** (ISBN: {clicked_isbn})")
                    
                    st.markdown("---")
                    st.markdown("### 📊 Top 10 sách được rerank:")
                    
                    # Hiển thị top 10 với debug info
                    for rank, (isbn, final_score) in enumerate(personalized[:10], 1):
                        if isbn in debug_info:
                            info = debug_info[isbn]
                            book = books_df[books_df['ISBN'] == isbn]
                            book_title = book.iloc[0]['Book-Title'] if not book.empty else isbn
                            
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**{rank}. {book_title}**")
                                st.caption(f"ISBN: {isbn}")
                            
                            with col2:
                                st.metric("Final Score", f"{final_score:.3f}")
                            
                            # Chi tiết boost
                            details = []
                            if info['cf_similarity'] > 0:
                                cf_detail = f"CF: {info['cf_similarity']:.3f} (+{info['cf_boost']:.3f})"
                                if info.get('use_weighted_sum') and info.get('cf_similarities_count', 0) > 1:
                                    cf_detail += f" [weighted sum từ {info['cf_similarities_count']} sách]"
                                elif info.get('best_cf_clicked'):
                                    cf_detail += f" [max từ {info['best_cf_clicked'][:12]}...]"
                                details.append(cf_detail)
                                
                                # Hiển thị chi tiết từng clicked book (nếu có nhiều hơn 1)
                                if info.get('all_cf_similarities') and len(info['all_cf_similarities']) > 1:
                                    cf_details_list = []
                                    for clicked_isbn, sim_info in info['all_cf_similarities'].items():
                                        recency_str = f", recency: {sim_info['recency_weight']:.2f}" if info.get('use_recency_weight') and sim_info['recency_weight'] < 1.0 else ""
                                        cf_details_list.append(f"{clicked_isbn[:8]}...: {sim_info['raw_similarity']:.2f}{recency_str}")
                                    st.caption(f"   📊 CF với từng clicked: {', '.join(cf_details_list)}")
                                    
                            if info['tfidf_similarity'] > 0:
                                tfidf_detail = f"TF-IDF: {info['tfidf_similarity']:.3f} (+{info['tfidf_boost']:.3f})"
                                if info.get('use_weighted_sum') and info.get('tfidf_similarities_count', 0) > 1:
                                    tfidf_detail += f" [weighted sum từ {info['tfidf_similarities_count']} sách]"
                                elif info.get('best_tfidf_clicked'):
                                    tfidf_detail += f" [max từ {info['best_tfidf_clicked'][:12]}...]"
                                details.append(tfidf_detail)
                                
                                # Hiển thị chi tiết TF-IDF
                                if info.get('all_tfidf_similarities') and len(info['all_tfidf_similarities']) > 1:
                                    tfidf_details_list = []
                                    for clicked_isbn, sim_info in info['all_tfidf_similarities'].items():
                                        recency_str = f", recency: {sim_info['recency_weight']:.2f}" if info.get('use_recency_weight') and sim_info['recency_weight'] < 1.0 else ""
                                        tfidf_details_list.append(f"{clicked_isbn[:8]}...: {sim_info['raw_similarity']:.2f}{recency_str}")
                                    st.caption(f"   📊 TF-IDF với từng clicked: {', '.join(tfidf_details_list)}")
                            
                            if details:
                                st.caption(" | ".join(details))
                            else:
                                st.caption("Không có boost (giữ nguyên score)")
                            
                            st.caption(f"Base score: {info['base_score']:.3f} → Final: {info['final_score']:.3f}")
                            st.markdown("---")

            # Giới hạn 10 items
            personalized = personalized[:10] if personalized else []

            if personalized:
                cols = st.columns(5)
                for idx, (isbn, pred_rating) in enumerate(personalized[:5]):
                    with cols[idx]:
                        display_book_card(isbn, books_df, reason='personalized', score=pred_rating,
                                          show_button=True, key_suffix=f"pers_{idx}")

                cols = st.columns(5)
                for idx, (isbn, pred_rating) in enumerate(personalized[5:10]):
                    with cols[idx]:
                        display_book_card(isbn, books_df, reason='personalized', score=pred_rating,
                                          show_button=True, key_suffix=f"pers_{idx + 5}")
    
    with tab2:
        show_data_analysis(books_df, users_df, ratings_df)


if __name__ == "__main__":
    main()