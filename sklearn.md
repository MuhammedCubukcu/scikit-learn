# scikit-learn Modülleri

## `datasets`
- `load_iris()`: Iris veri setini yükler.
- `load_digits()`: El yazısı rakamlar veri setini yükler.
- `fetch_20newsgroups()`: 20 gruplu haber grubu veri setini yükler.

## `model_selection`
- `train_test_split()`: Veri setini eğitim ve test setlerine ayırır.
- `GridSearchCV()`: Grid araması ile model hiperparametrelerini optimize eder.
- `KFold()`: K-katlı çapraz doğrulama sağlar.
- `cross_val_score()`: Çapraz doğrulama skorlarını hesaplar.

## `linear_model`
- `LinearRegression()`: Lineer regresyon modeli oluşturur.
- `LogisticRegression()`: Lojistik regresyon modeli oluşturur.
- `Ridge()`: Ridge regresyon modeli oluşturur.
- `Lasso()`: Lasso regresyon modeli oluşturur.
- `SGDClassifier()`: Stochastic Gradient Descent (SGD) sınıflandırıcısı oluşturur.
- `SGDRegressor()`: Stochastic Gradient Descent (SGD) regresyon modeli oluşturur.

## `tree`
- `DecisionTreeClassifier()`: Karar ağacı sınıflandırıcı oluşturur.
- `DecisionTreeRegressor()`: Karar ağacı regresyon modeli oluşturur.

## `ensemble`
- `RandomForestClassifier()`: Rastgele orman sınıflandırıcısı oluşturur.
- `GradientBoostingClassifier()`: Gradyan artırma sınıflandırıcısı oluşturur.
- `GradientBoostingRegressor()`: Gradyan artırma regresyon modeli oluşturur.
- `AdaBoostClassifier()`: AdaBoost sınıflandırıcısı oluşturur.
- `BaggingClassifier()`: Bagging sınıflandırıcısı oluşturur.
- `ExtraTreesClassifier()`: Extra Trees sınıflandırıcısı oluşturur.
- `VotingClassifier()`: Oy verme sınıflandırıcısı oluşturur.
- `StackingClassifier()`: Yığıtlanmış sınıflandırıcı oluşturur.

## `neighbors`
- `KNeighborsClassifier()`: K-En Yakın Komşular sınıflandırıcısı oluşturur.
- `RadiusNeighborsClassifier()`: Radius tabanlı komşu sınıflandırıcısı oluşturur.

## `svm`
- `SVC()`: Destek vektör makinesi sınıflandırıcı oluşturur.
- `SVR()`: Destek vektör makinesi regresyon modeli oluşturur.
- `LinearSVC()`: Doğrusal destek vektör makinesi sınıflandırıcı oluşturur.
- `OneClassSVM()`: Tek sınıflı destek vektör makinesi oluşturur.

## `naive_bayes`
- `MultinomialNB()`: Çoklu nominal (discrete) Bayes sınıflandırıcısı oluşturur.
- `GaussianNB()`: Gaussian Naive Bayes sınıflandırıcısı oluşturur.

## `cluster`
- `KMeans()`: K-means kümeleme algoritması uygular.
- `AgglomerativeClustering()`: Hiyerarşik kümeleme yapar.
- `DBSCAN()`: DBSCAN kümeleme algoritması uygular.

## `manifold`
- `TSNE()`: T-distributed Stochastic Neighbor Embedding (t-SNE) yapar.
- `Isomap()`: İzometrik haritalama yapar.
- `LocallyLinearEmbedding()`: Yerel doğrusal gömme yapar.

## `decomposition`
- `PCA()`: Temel Bileşen Analizi (PCA) yapar.
- `NMF()`: Non-negative Matrix Factorization (NMF) yapar.
- `TruncatedSVD()`: Truncated Singular Value Decomposition (SVD) yapar.

## `preprocessing`
- `StandardScaler()`: Veriyi standartlaştırır.
- `OneHotEncoder()`: Kategorik değişkenleri sayısal forma dönüştürür.
- `MinMaxScaler()`: Veriyi 0 ile 1 aralığına dönüştürür.
- `SimpleImputer()`: Eksik verileri doldurur.
- `KNNImputer()`: K-en yakın komşu ile eksik verileri doldurur.
- `IterativeImputer()`: İteratif olarak eksik verileri doldurur.
- `LabelEncoder()`: Etiketleri sayısal forma dönüştürür.
- `PolynomialFeatures()`: Polinom özellikler ekler.

## `metrics`
- `accuracy_score()`: Sınıflandırma doğruluğunu hesaplar.
- `confusion_matrix()`: Karmaşıklık matrisini oluşturur.
- `mean_squared_error()`: Ortalama karesel hata hesaplar.
- `r2_score()`: R-kare skoru hesaplar.
- `mean_absolute_error()`: Ortalama mutlak hata hesaplar.
- `precision_score()`: Kesinlik değerini hesaplar.
- `log_loss()`: Logarithmic loss'u hesaplar.
- `f1_score()`: F1 skoru hesaplar.
- `precision_recall_curve()`: Kesinlik-duyarlılık eğrisi oluşturur.

## `impute`
- `SimpleImputer()`: Eksik verileri doldurur.
- `KNNImputer()`: K-en yakın komşu ile eksik verileri doldurur.
- `IterativeImputer()`: İteratif olarak eksik verileri doldurur.