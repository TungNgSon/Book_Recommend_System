"""
Script tính item-item similarity matrix (ULTRA OPTIMIZED)
Chỉ lưu Top-N similar items thay vì toàn bộ matrix
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import pickle
from pathlib import Path
from tqdm import tqdm

def compute_item_similarity_topn(min_ratings=10, top_n=50):
    """
    Tính cosine similarity nhưng CHỈ LƯU TOP-N items tương tự nhất
    Tiết kiệm memory cực kỳ nhiều

    Args:
        min_ratings: Chỉ giữ books có ít nhất N ratings
        top_n: Chỉ lưu N items tương tự nhất cho mỗi item
    """
    print("📚 Đang load dữ liệu...")

    # Load ratings data
    ratings_path = Path('../dataset/cleaned/Ratings_cleaned.csv')
    ratings = pd.read_csv(ratings_path)

    print(f"✅ Loaded {len(ratings):,} ratings")
    print(f"📖 Số books: {ratings['ISBN'].nunique():,}")
    print(f"👥 Số users: {ratings['User-ID'].nunique():,}")

    # Filter: Chỉ giữ books có đủ ratings
    print(f"\n🔧 Filtering books với ít nhất {min_ratings} ratings...")
    book_counts = ratings['ISBN'].value_counts()
    popular_books = book_counts[book_counts >= min_ratings].index.tolist()
    ratings_filtered = ratings[ratings['ISBN'].isin(popular_books)]

    print(f"✅ Còn lại {len(ratings_filtered):,} ratings")
    print(f"📖 Còn lại {ratings_filtered['ISBN'].nunique():,} books")

    # Tạo user-item matrix
    print("\n🔧 Đang tạo user-item matrix...")
    user_item_matrix = ratings_filtered.pivot_table(
        index='User-ID',
        columns='ISBN',
        values='Book-Rating',
        fill_value=0
    )

    print(f"📊 Matrix shape: {user_item_matrix.shape}")

    # Convert to sparse matrix
    print("\n💾 Converting to sparse matrix...")
    sparse_matrix = csr_matrix(user_item_matrix.values)
    isbns = user_item_matrix.columns.tolist()
    n_items = len(isbns)

    print(f"✅ Sparse matrix: {sparse_matrix.shape}")
    print(f"💾 Sparsity: {100 * (1 - sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1])):.2f}%")

    # Tính similarity từng item một và chỉ lưu top-N
    print(f"\n🧮 Đang tính top-{top_n} similar items cho mỗi book...")
    print("⏳ Đây sẽ mất vài phút, vui lòng đợi...")

    # Dictionary để lưu top-N similarities
    # Format: {isbn: [(similar_isbn, score), ...]}
    similarity_dict = {}

    # Tính theo batch nhỏ để tránh memory overflow
    batch_size = 100

    for i in range(0, n_items, batch_size):
        end = min(i + batch_size, n_items)
        print(f"   Processing items {i+1} to {end} / {n_items}...")

        # Tính similarity cho batch này với TẤT CẢ items
        batch_similarity = cosine_similarity(
            sparse_matrix[:, i:end].T,  # Batch hiện tại
            sparse_matrix.T              # Toàn bộ items
        )

        # Với mỗi item trong batch, lưu top-N similar items
        for j, isbn_idx in enumerate(range(i, end)):
            isbn = isbns[isbn_idx]
            scores = batch_similarity[j]

            # Lấy indices của top-N (bỏ chính nó)
            # argsort trả về indices, [::-1] để reverse (cao nhất trước)
            top_indices = np.argsort(scores)[::-1][1:top_n+1]  # Bỏ index 0 (chính nó)

            # Lưu (isbn, score) pairs
            similar_items = [
                (isbns[idx], float(scores[idx]))
                for idx in top_indices
                if scores[idx] > 0.01  # Chỉ lưu nếu similarity > threshold
            ]

            similarity_dict[isbn] = similar_items

    print(f"\n✅ Đã tính similarity cho {len(similarity_dict):,} books")

    # Lưu file
    print("\n💾 Đang lưu file...")
    output_path = Path('../dataset/cleaned/item_similarity.pkl')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(similarity_dict, f)

    print(f"✅ Đã lưu similarity dictionary tại: {output_path}")
    print(f"📦 File size: {output_path.stat().st_size / (1024*1024):.2f} MB")

    # Test thử
    print("\n🧪 Testing similarity dictionary...")
    sample_isbn = list(similarity_dict.keys())[0]
    similar_books = similarity_dict[sample_isbn]
    print(f"\nTop 5 books tương tự với ISBN {sample_isbn}:")
    for isbn, score in similar_books[:5]:
        print(f"  {isbn}: {score:.4f}")

    # Statistics
    avg_similar = np.mean([len(v) for v in similarity_dict.values()])
    print(f"\n📊 Thống kê:")
    print(f"   Trung bình mỗi book có {avg_similar:.1f} similar items")

    return similarity_dict

if __name__ == "__main__":
    print("="*60)
    print("🚀 TÍNH ITEM-ITEM SIMILARITY (ULTRA OPTIMIZED)")
    print("="*60)
    print("\n💡 Phiên bản này CHỈ LƯU TOP-N similar items")
    print("   thay vì toàn bộ matrix → tiết kiệm memory cực nhiều!")

    # Hỏi user về parameters
    print("\n⚙️  Cấu hình:")
    print("   - min_ratings: Chỉ giữ books có ít nhất N ratings")
    print("   - top_n: Số lượng similar items lưu cho mỗi book")

    min_ratings_input = input("\nNhập min_ratings [mặc định: 10]: ").strip()
    min_ratings = int(min_ratings_input) if min_ratings_input else 10

    top_n_input = input("Nhập top_n [mặc định: 50]: ").strip()
    top_n = int(top_n_input) if top_n_input else 50

    print(f"\n📌 Cấu hình: min_ratings={min_ratings}, top_n={top_n}")

    try:
        similarity_dict = compute_item_similarity_topn(
            min_ratings=min_ratings,
            top_n=top_n
        )

        print("\n" + "="*60)
        print("✅ HOÀN THÀNH!")
        print("="*60)
        print("\n📌 File đã tạo: ../dataset/cleaned/item_similarity.pkl")
        print("📌 Bây giờ bạn có thể chạy: streamlit run app.py")
        print("\n⚠️  LƯU Ý: File similarity giờ là DICTIONARY, không phải DataFrame")
        print("   Nhưng code trong app.py và utils.py vẫn hoạt động bình thường!")

    except MemoryError:
        print("\n" + "="*60)
        print("❌ VẪN HẾT MEMORY!")
        print("="*60)
        print("\n💡 Giải pháp:")
        print("1. Tăng min_ratings lên 20 hoặc 30")
        print("2. Giảm top_n xuống 20 hoặc 30")
        print("3. Chạy trên máy RAM cao hơn hoặc Google Colab")

    except Exception as e:
        print(f"\n❌ LỖI: {e}")
        import traceback
        traceback.print_exc()