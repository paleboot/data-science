import re
import string
import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer

# initialize documents

d1 = 'Virus yang telah menjadi pandemi global ini telah menginfeksi 177 negara. Hingga Kamis (19/3/2020) tercatat 25 orang dari 309 kasus di Indonesia dinyatakan meninggal dunia. Sementara itu secara global virus yang berasal dari Wuhan, China ini telah menginfeksi 226.838 pasien. Gejala awal virus ini bisa bermula dari demam, kelelahan hingga sesak nafas.'


d2 = 'Gejala virus Corona covid-19 sangat penting untuk kita kenali dan ketahui. Virus ini telah ditetapkan sebagai pandemi global. Organisasi Kesehatan Dunia (WHO) telah menyatakan darurat kesehatan masyarakat terkait virus ini.  Virus Corona covid-19 ini cepat menyebar dan menjangkit banyak korban. Sekilas, gejala virus Corona mirip dengan flu atau pilek biasa. Faktanya, COVID-19 jauh lebih mematikan daripada flu musiman. Gejala virus Corona berkisar dari gejala ringan hingga parah dan kematian.'


d3 = 'Dilansir dari laman kesehatan web.md, gejala umum yang dirasakan oleh pasien positif coronavirus adalah demam, batuk, dan kesulitan bernafas. Gejala ini memang kurang lebih sama dengan gejala seseorang terkena demam atau flu. Tetapi, virus corona bisa membawa pasien ke tingkat yang lebih parah, yakni pneumonia atau bahkan terganggunya saluran pernafasan. Semua gejala ini kemungkinan akan terjadi selama 2 hari atau paling lama 2 minggu'


d4 = '"Infeksi yang menyerang tidak diketahui (tidak memiliki gejala) sehingga mendorong penyebaran wabah," kata rekan penulis studi Jeffrey Shaman dari Columbia University Mailman School. "Sebagian besar infeksi dalam fase ringan dan hanya ada sedikit gejala yang muncul, bahkan tidak ada. Banyak orang tidak sadar telah terinfeksi SARS-CoV-2, kebanyakan mengira hanya flu ringan," ujar Shaman.'

# case folding

d1 = d1.lower()
d2 = d2.lower()
d3 = d3.lower()
d4 = d4.lower()

# filtering (number followed by symbol)

d1_filter = re.sub(r'\d+', "", d1)
d1_filter = d1_filter.translate(str.maketrans("","",string.punctuation))

d2_filter = re.sub(r'\d+', "", d2)
d2_filter = d2_filter.translate(str.maketrans("","",string.punctuation))

d3_filter = re.sub(r'\d+', "", d3)
d3_filter = d3_filter.translate(str.maketrans("","",string.punctuation))

d4_filter = re.sub(r'\d+', "", d4)
d4_filter = d4_filter.translate(str.maketrans("","",string.punctuation))

# stop word removal

stopword_factory = StopWordRemoverFactory()
stopword = stopword_factory.create_stop_word_remover()

d1_filter = stopword.remove(d1_filter)
d2_filter = stopword.remove(d2_filter)
d3_filter = stopword.remove(d3_filter)
d4_filter = stopword.remove(d4_filter)

# stemming

stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

d1_terms = stemmer.stem(d1_filter)
d2_terms = stemmer.stem(d2_filter)
d3_terms = stemmer.stem(d3_filter)
d4_terms = stemmer.stem(d4_filter)

# print(d1_terms, "\n")
# print(d2_terms,"\n")
# print(d3_terms, "\n")
# print(d4_terms, "\n")

# tf-idf calculation

documents = [d1_terms, d2_terms, d3_terms, d4_terms]
tfidf = TfidfVectorizer()

features = tfidf.fit_transform(documents)
df = pd.DataFrame(features.todense(),columns=tfidf.get_feature_names())

print(df[['gejala', 'nafas']])