import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('full_data.csv')

""" Satlik yorum yapabilmek için zaman sütununu tarih saate dönüştürün ve yılı ve ayı çıkarın"""
data['time'] = pd.to_datetime(data['time'], format='%d:%m:%Y:%H:%M')
data['year'] = data['time'].dt.year
data['month'] = data['time'].dt.month

""" yenilenebilir enerji kaynaklarını birleştirdiğim yer"""
data['toplam_yenilenebilir_üretim'] = data['wind'] + data['solar'] + data['hydro_dam'] + data['hydro_river']

"""zaman sütununu ayarlıyoruz"""
data_numeric = data.drop(columns=['time'])

"""Radyal temel fonksiyonu"""
def rbf(x, c, s):
    return np.exp(-np.linalg.norm(x - c) ** 2 / (2 * s ** 2))

"""rbf tahmin fonksiyonu"""
def rbf_predict(X, centers, sigma, W):
    R = np.zeros((X.shape[0], len(centers)))
    for i in range(X.shape[0]):
        for j in range(len(centers)):
            R[i, j] = rbf(X[i], centers[j], sigma)
    return np.dot(R, W)

"""Özellikleri ölçeklendirme"""
scaler = StandardScaler()
scaled_data_numeric = pd.DataFrame(scaler.fit_transform(data_numeric), columns=data_numeric.columns)

"""Tahmin edilecek sütunlar"""
target_columns = ['tüketim_MWh', 'toplam_üretim_MWh', 'toplam_yenilenebilir_üretim', 'TRY/MWh']

"""Sonuçlar için bir sözlük oluşturma"""
results = {}

"""K-means ile optimal küme sayısını bulma fonksiyonu"""
def find_optimal_clusters(X):
    distortions = []
    max_clusters = min(15, len(X))
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
        distortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

    return max(1, distortions.index(min(distortions)) + 1)

for target in target_columns:
    X = scaled_data_numeric.drop(columns=[target]).values
    y = scaled_data_numeric[target].values

    """Eğitim ve test veri setlerini oluşturma"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    """Optimal küme sayısını belirleme"""
    optimal_clusters = find_optimal_clusters(X_train)

    """K-means ile merkezleri hesaplama"""
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42).fit(X_train)
    centers = kmeans.cluster_centers_

    """Sigma (yayılma parametresi) hesaplama"""
    d_max = np.max(cdist(centers, centers, 'euclidean'))
    sigma = d_max / np.sqrt(2 * len(centers))

    """Eğitim verisi için RBF tabakası çıktısını hesaplama"""
    R_train = np.zeros((X_train.shape[0], len(centers)))
    for i in range(X_train.shape[0]):
        for j in range(len(centers)):
            R_train[i, j] = rbf(X_train[i], centers[j], sigma)

    """Çıkış ağırlıklarını hesaplama"""
    W = np.dot(np.linalg.pinv(R_train), y_train)

    """Test verisi için RBF tabakası çıktısını hesaplama"""
    R_test = np.zeros((X_test.shape[0], len(centers)))
    for i in range(X_test.shape[0]):
        for j in range(len(centers)):
            R_test[i, j] = rbf(X_test[i], centers[j], sigma)
    y_pred_scaled = rbf_predict(X_test, centers, sigma, W)

    """Sadece hedef sütun için scaler oluşturup, tahminleri orijinal ölçeğe geri çevirme"""
    target_scaler = StandardScaler()
    target_scaler.fit(data_numeric[[target]])
    y_pred_original = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_pred = rbf_predict(X_test, centers, sigma, W)
    y_test_original = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    """Performans metriklerini hesaplama"""
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test_original, y_pred_original)

    """Başarı yüzdesini hesaplama"""
    success_percentage = r2 * 100

    results[target] = {
        'Tahmini Değer': y_pred_original[0] if len(y_pred_original) > 0 else None,
        'Ortalama Kare Hatası': mse,
        'Başarı Yüzdesi (%)': success_percentage
    }

results_df = pd.DataFrame.from_dict(results, orient='index')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(results_df)
