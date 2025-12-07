import numpy as np
from sklearn.decomposition import PCA
from .helper import decode_image, encode_image

def process_pca(contents, k):
    """
    Tái tạo ảnh dùng PCA (Scikit-learn) với k thành phần chính.
    """
    img = decode_image(contents)
    if img is None: return None, 0, [], "0 KB"

    # lấy dài và rộng
    h, w = img.shape
    
    # Đảm bảo k không vượt quá min(height, width)
    k = min(k, h, w)

    # 1. Khởi tạo và Fit mô hình PCA 
    pca = PCA(n_components=k)
    
    # Transform: Nén ảnh (Giảm chiều dữ liệu)
    img_transformed = pca.fit_transform(img)
    
    # Inverse Transform: Tái tạo ảnh từ các thành phần chính [cite: 17]
    img_reconstructed = pca.inverse_transform(img_transformed)
    img_reconstructed = np.clip(img_reconstructed, 0, 255).astype(np.uint8)

    # 2. Lấy thông tin phương sai từ model PCA
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    retention = cumulative_variance[-1] * 100  # Giá trị cuối cùng là tổng tích lũy của k thành phần
    
    # 3. Ước lượng dung lượng
    # PCA lưu: Mean vector + Components (Eigenvectors) + Transformed data
    compressed_size = (img_transformed.size + pca.components_.size + pca.mean_.size) * 4
    size_str = f"{compressed_size/1024:.2f} KB"

    return encode_image(img_reconstructed), retention, cumulative_variance, size_str