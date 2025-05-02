import pandas as pd
import re 
import string 
# import nltk
from nltk.corpus import stopwords

# nltk.download('punkt')
# nltk.download('stopwords')

days_of_week = {'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun', 
                'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'}
months = {'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
          'january', 'february', 'march', 'april', 'may', 'june', 'july', 
          'august', 'september', 'october', 'november', 'december'}
kamus_normalisasi = {
    'yg': 'yang',
    'utk': 'untuk',
    'dgn': 'dengan',
    'org': 'orang',
    'krn': 'karena',
    'spt': 'seperti',
    'jd': 'jadi',
    'tdk': 'tidak',
    'pd': 'pada',
    'dr': 'dari',
    'dlm': 'dalam',
    'ttg': 'tentang',
    'tsb': 'tersebut',
    'sm': 'sama',
    'bs': 'bisa',
    'jg': 'juga',
    'sdh': 'sudah',
    'sy': 'saya',
    'gk': 'tidak',
    'ga': 'tidak',
    'gak': 'tidak',
    'klo': 'kalau',
    'kalo': 'kalau',
    'bgt': 'banget',
    'aja': 'saja',
    'tp': 'tapi',
    'trs': 'terus',
    'trz': 'terus',
    'bkn': 'bukan',
    'hrs': 'harus',
    'udh': 'sudah',
    'dpt': 'dapat',
    'skrg': 'sekarang',
    'sblm': 'sebelum',
    'stelah': 'setelah',
    'stlh': 'setelah',
    'jln': 'jalan',
    'thn': 'tahun',
    'bln': 'bulan',
    'mgkn': 'mungkin',
    'bgtu': 'begitu',
    'pnh': 'pernah',
    'sngt': 'sangat',
    'kmrn': 'kemarin',
    'spt': 'seperti',
    'sprti': 'seperti',
    'jgn': 'jangan',
    'kpn': 'kapan',
    'knp': 'kenapa',
    'kyk': 'kayak',
    'kya': 'kayak',
    'gmn': 'bagaimana',
    'gimana': 'bagaimana',
    'dgr': 'dengar',
    'lbh': 'lebih',
    'bnyk': 'banyak',
    'msk': 'masuk',
    'tdk': 'tidak',
    'sblm': 'sebelum',
    'blm': 'belum',
    'bru': 'baru',
    'bhw': 'bahwa',
    'msh': 'masih',
    'wkwk': 'tertawa',
    'wkwkwk': 'tertawa',
    'haha': 'tertawa',
    'hehe': 'tertawa',
    'hmm': 'hmm',
    'sih': 'sih',
    'lho': 'lho',
    'loh': 'loh',
    'kok': 'kok',
    'dong': 'dong',
    'deh': 'deh',
    'gue': 'saya',
    'lu': 'kamu',
    'loe': 'kamu',
    'elo': 'kamu',
    'kmu': 'kamu',
    'aq': 'saya',
    'ak': 'saya',
    'aku': 'saya',
    'w': 'saya',
    'gw': 'saya',
    'lo': 'kamu',
    'lg': 'lagi',
    'at': 'atau',
    'mslh': 'masalah',
    'smua': 'semua',
    'sja': 'saja',
    'sdikit': 'sedikit',
    'kalo': 'kalau',
    'karna': 'karena',
    'emang': 'memang',
    'gini': 'begini',
    'gitu': 'begitu',
    'denger': 'dengar',
    'bikin': 'membuat',
    'bilang': 'mengatakan',
    'ngga': 'tidak',
    'pengen': 'ingin',
    'mau': 'ingin',
    'biar': 'agar',
    'tau': 'tahu',
    'udah': 'sudah',
    'enggak': 'tidak',
    'ngomong': 'berbicara',
    'nggak': 'tidak',
    'gapapa': 'tidak apa-apa',
    'gaada': 'tidak ada',
    'liat': 'lihat',
    'ngeliat': 'melihat',
    'doang': 'saja',
}

try:
    indonesian_stopwords = set(stopwords.words('indonesian'))
except:
    indonesian_stopwords = set()
    print("Warning: NLTK Indonesian stopwords not found, using custom stopwords list only")

additional_stopwords = {
    'yah', 'iya', 'sih', 'nya', 'nih', 'ah', 'oh', 'eh',
    'juga', 'dan', 'itu', 'ini', 'yang', 'di', 'ke', 'ya',
    'ada', 'tidak', 'dengan', 'saya', 'kamu', 'kami', 'kita',
    'untuk', 'pada', 'adalah', 'dari', 'dalam', 'akan', 'oleh',
    'apa', 'siapa', 'kapan', 'dimana', 'bagaimana', 'mengapa', 'kenapa',
    'atau', 'seperti', 'jadi', 'jika', 'kalau', 'karena', 'sebab',
    'lalu', 'kemudian', 'tetapi', 'namun', 'melainkan', 'meskipun',
    'sedangkan', 'padahal', 'selama', 'sementara', 'setelah', 'sebelum',
    'sejak', 'sampai', 'ketika', 'saat', 'waktu', 'seraya', 'sambil',
    'agar', 'supaya', 'biar', 'maka', 'bahwa', 'asalkan', 'bila',
    'demi', 'guna', 'hingga', 'tanpa', 'kecuali', 'selain',
    'sebagai', 'tentang', 'terhadap', 'menurut', 'berdasarkan',
    'melalui', 'lewat', 'sesuai', 'yakni',
}
indonesian_stopwords.update(additional_stopwords)

def clean_text(text):
    # Skip cleaning if text is not a string (e.g., NaN)
    if not isinstance(text, str):
        return text
        
    # Menghapus URL
    text = re.sub(r'http\S+', '', text)
    # Menghapus username
    text = re.sub(r'@\w+', '', text)
    # Menghapus hashtag
    text = re.sub(r'#\w+', '', text)
    # Menghapus emoji (range Unicode untuk emoji)
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251" 
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    # Menghapus tanda baca
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Menghapus angka
    text = re.sub(r'\d+', '', text)
    # Menghapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    # Mengubah ke lowercase
    text = text.lower()
    
    return text
def combine_lexicon():
    custom_lexicon = pd.read_csv("./lexicon.csv")
    default_lexicon = pd.read_csv("./default_lexicon.csv")
    combined = pd.concat([default_lexicon, custom_lexicon])
    combined = combined.drop_duplicates(subset=["word"], keep="last")
    combined.set_index("word", inplace=True)
    return combined

list_emotion = {
    "1": "Senang",
    "2": "Percaya",
    "3": "Terkejut",
    "4": "Netral",
    "5": "Takut",
    "6": "Sedih",
    "7": "Marah"
}

reverse_emotion = {
    "1": "7",
    "2": "5",
    "3": "3",
    "4": "4",
    "5": "2",
    "6": "1",
    "7": "1"
}

def hapus_hari_bulan(text):
    """Menghapus kata hari dan bulan di awal teks"""
    if not isinstance(text, str):
        return text
    
    words = text.split()
    if len(words) >= 2:
        # Periksa apakah 2 kata pertama adalah hari dan bulan
        first_word_lower = words[0].lower()
        second_word_lower = words[1].lower()
        
        # Jika kata pertama adalah nama hari dan kata kedua adalah nama bulan
        if first_word_lower in days_of_week and second_word_lower in months:
            # Hapus dua kata pertama
            return ' '.join(words[2:])
    
    return text

def normalisasi(tokens):
    """Mengubah kata-kata tidak baku menjadi baku menggunakan kamus."""
    return [kamus_normalisasi.get(word, word) for word in tokens]

def hapus_stopwords(tokens):
    """Menghapus stopwords dari daftar token."""
    return [word for word in tokens if word not in indonesian_stopwords]



def get_emotion_from_dataset(word):
    total = {}
    dataset = pd.read_csv("./data.csv", index_col="tweet")
    # Iterasi melalui setiap kalimat (indeks) dalam dataset
    for sentence in dataset.index:
        # Pecah kalimat menjadi kata-kata
        words = sentence.lower().split()
        if word.lower() in words:
            # Ambil data emosi untuk kalimat ini
            row = dataset.loc[sentence]
            emotion = str(row["emotion"])
            count = total.get(emotion, 0)
            total[emotion] = count + 1
    max_emotion = "4"
    max_count = 0
    
    if total:  # Pastikan dictionary tidak kosong
        for emotion, count in total.items():
            if count > max_count:
                max_count = count
                max_emotion = emotion
    
    return max_emotion

def get_emotion(word):
    lx = combine_lexicon()
    try:
        emotion = lx.loc[word]
        return str(emotion["emotion"])
    except KeyError:
        max_emotion = get_emotion_from_dataset(word)
        
        if max_emotion:
            new_row = pd.Series({"emotion": max_emotion}, name=word)
            
            lx = pd.concat([lx, pd.DataFrame([new_row])])
            
            lx.to_csv("./lexicon.csv", index_label="word")
            
            return max_emotion
        else:
            return "4" 

def predict_word(word):
    proccess_test = clean_text(word)

    proccess_test = hapus_hari_bulan(proccess_test)

    split_proccess_text = proccess_test.split(" ")

    split_proccess_text = normalisasi(split_proccess_text)

    max_len = len(split_proccess_text)

    total = {}

    index = 0
    # Daftar kata negasi
    negation_words = ["tidak", "tdk", "tak", "bukan", "jangan"]
    
    # Daftar kata intensifier
    intensifier_words = ["sangat", "sgt", "amat", "sekali", "banget"]
    desc = {}
    while index < max_len:
        word = split_proccess_text[index]
        
        # Case 1: Intensifier + Negasi (contoh: "sangat tidak baik")
        if word in intensifier_words and index + 2 < max_len and split_proccess_text[index + 1] in negation_words:
            w = split_proccess_text[index + 2]  # Kata setelah "sangat tidak"
            next_emotion = get_emotion(w)
            r_emotion = reverse_emotion[next_emotion]  # Membalikkan emosi karena ada "tidak"
            wi = split_proccess_text[index + 1]

            key = word + ' ' + wi + ' ' + w
            # Karena ada "sangat", kita tambahkan bobot lebih
            count = total.get(r_emotion, 0)
            total[r_emotion] = count + 3  # Bobot lebih tinggi karena intensifikasi
            desc[key] = r_emotion
            index += 3  # Lompat 3 kata
            continue
            
        # Case 2: Negasi (contoh: "tidak baik")
        elif word in negation_words and index + 1 < max_len:
            w = split_proccess_text[index + 1]
            next_emotion = get_emotion(w)
            r_emotion = reverse_emotion[next_emotion]
            key = word + " " + w
            if r_emotion == next_emotion:  # Jika emosi tidak berubah saat dibalik (netral)
                count = total.get("4", 0)
                desc[key] = "4"
                total["4"] = count + 1
            else:
                desc[key] = r_emotion
                count = total.get(r_emotion, 0)
                total[r_emotion] = count + 2  # Bobot 2 untuk negasi
            
            index += 2  # Lompat 2 kata
            continue
            
        # Case 3: Intensifier (contoh: "sangat baik")
        elif word in intensifier_words and index + 1 < max_len:
            w = split_proccess_text[index + 1]
            emotion = get_emotion(w)
            count = total.get(emotion, 0)
            total[emotion] = count + 2  # Bobot 2 untuk intensifikasi
            key = word + " " + w
            desc[key] = emotion

            index += 2  # Lompat 2 kata
            continue
            
        # Case 4: Kata biasa
        else:
            emotion = get_emotion(word)
            count = total.get(emotion, 0)
            total[emotion] = count + 1  # Bobot normal
            desc[word] = emotion
            index += 1  # Lompat 1 kata
    
    # Menentukan emosi dominan
    dominant_emotion = max(total, key=total.get)
    return dominant_emotion, desc

def predic_dataset():
    """
    Membaca dataset, menghapus duplikat, menyimpan tweet unik,
    dan memprediksi emosi untuk setiap tweet.
    Menyimpan hasil ke dataset_predicted.csv
    """
    print("Membaca dataset...")
    dataset = pd.read_csv("./data.csv")
    
    # Menampilkan informasi awal
    total_awal = len(dataset)
    print(f"Jumlah total tweet: {total_awal}")
    
    # Hapus duplikat berdasarkan kolom 'tweet'
    dataset_unique = dataset.drop_duplicates(subset=['tweet'])
    tweets_removed = total_awal - len(dataset_unique)
    print(f"Menghapus {tweets_removed} tweet duplikat")
    print(f"Menyisakan {len(dataset_unique)} tweet unik")

    # Simpan tweet unik sebelum prediksi
    dataset_unique.to_csv("./data.csv", index=False)
    print("Tweet unik disimpan ke dataset_unique.csv")

    # Set tweet sebagai index untuk prediksi
    dataset_unique.set_index('tweet', inplace=True)

    # Tambahkan kolom prediksi
    dataset_unique['predict'] = "4"  # Default netral
    
    print("Memulai prediksi emosi...")
    for i, tweet in enumerate(dataset_unique.index.values):
        try:
            # Prediksi emosi menggunakan fungsi eksternal
            emotion, _ = predict_word(tweet)
            # Simpan hasil prediksi
            dataset_unique.at[tweet, 'predict'] = emotion
            
            if (i + 1) % 50 == 0:
                print(f"Memproses tweet {i+1}/{len(dataset_unique)}")
        
        except Exception as e:
            print(f"Error pada tweet {i+1}: {str(e)}")
            print(e)
    
    # Simpan hasil prediksi
    dataset_unique.reset_index().to_csv("./dataset_predicted.csv", index=False)
    print("Prediksi selesai! Hasil disimpan ke dataset_predicted.csv")