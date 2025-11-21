import csv
import numpy as np
from sklearn.cluster import KMeans
import requests

"""System rekomendacji filmów przy użyciu korelacji Pearsona

Skrypt nalezy wywołać za pomocą komendy: py GRA_AI_3.py
po czym w konsoli wpisac Nr uzytkownika z listy dla którego chcemy wykonać rekomendację

Wymagania: pip install numpy requests sklearn

Autorzy: Adrian Kopczyński, Gabriel Francke"""


def load_csv(path):
    """
    Wczytuje dane z pliku CSV

    Args:
        path (str): Ścieżka do pliku CSV

    Returns:
        users (dict): {użytkownik: {film: ocena}}
        movies (list): lista wszystkich filmów
    """
    users = {}      # {użytkownik: {film: ocena}}
    movies = set()  # lista wszystkich filmów

    with open(path, encoding="utf8") as f:
        reader = csv.reader(f)
        for row in reader:
            # row wygląda: [ , Imię Nazwisko, Film1, Ocena1, Film2, Ocena2, ...]
            if len(row) < 3:
                continue

            name = row[1].strip()
            ratings = {}

            for i in range(2, len(row), 2):
                if i + 1 >= len(row):
                    break
                film = row[i].strip()
                if not film:
                    continue
                try:
                    score = float(row[i + 1])
                except Exception:
                    continue
                ratings[film] = score
                movies.add(film)

            users[name] = ratings

    return users, sorted(list(movies))


def build_disliked(users, threshold=2):
    """
    Tworzy słownik filmów, których użytkownik raczej nie lubi
    na podstawie niskich ocen (<= threshold).

    Args:
        users (dict): {użytkownik: {film: ocena}}
        threshold (float): próg oceny uznawanej za dislike

    Returns:
        disliked (dict): {użytkownik: set(filmy)}
    """
    disliked = {}
    for user, ratings in users.items():
        disliked[user] = {film for film, score in ratings.items() if score <= threshold}
    return disliked


def build_rating_matrix(users, movies, user_list):
    """
    Tworzy macierz ocen użytkownik × film, brak oceny = 0.

    Args:
        users (dict): {użytkownik: {film: ocena}}
        movies (list): lista wszystkich filmów
        user_list (list): lista użytkowników w kolejności

    Returns:
        np.ndarray: macierz ocen (użytkownicy × filmy)
    """
    matrix = []
    for user in user_list:
        row = []
        for movie in movies:
            row.append(users[user].get(movie, 0))
        matrix.append(row)
    return np.array(matrix)


def find_optimal_k(rating_matrix, k_min=2, k_max=10):
    """
    Wyznacza optymalną liczbę klastrów metodą Elbow (największy spadek inercji).

    Args:
        rating_matrix (np.ndarray): macierz użytkownik x film
        k_min (int): minimalna liczba klastrów
        k_max (int): maksymalna liczba klastrów

    Returns:
        best_k (int): optymalna liczba klastrów
        inertias (list): lista inercji dla każdego k
    """
    inertias = []

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(rating_matrix)
        inertias.append(kmeans.inertia_)

    # obliczamy różnice (spadki) inercji
    drops = []
    for i in range(1, len(inertias)):
        drops.append(inertias[i - 1] - inertias[i])

    best_idx = np.argmax(drops)
    best_k = best_idx + 2   # bo drops zaczyna się od K=3

    return best_k, inertias


def recommend(users, disliked, movies, clusters, labels, user_list, target_user, top_k=5):
    """
    Generuje rekomendacje filmów dla użytkownika na podstawie klastrów
    i ocen innych użytkowników w tym samym klastrze.

    Args:
        users (dict): {użytkownik: {film: ocena}}
        disliked (dict): {użytkownik: set(filmy)}
        movies (list): lista wszystkich filmów
        clusters (dict): {klaster: [użytkownicy]}
        labels (np.ndarray): etykiety klastrów KMeans
        user_list (list): lista użytkowników
        target_user (str): użytkownik dla którego rekomendujemy
        top_k (int): ile filmów zwrócić

    Returns:
        list: [(film, średnia_ocena), ...] posortowane malejąco
    """
    if target_user not in users:
        return []

    idx = user_list.index(target_user)
    user_cluster = labels[idx]
    neighbours = clusters.get(user_cluster, [])

    seen = set(users[target_user].keys())
    bad = disliked.get(target_user, set())

    score_sum = {}
    score_count = {}

    for neighbour in neighbours:
        if neighbour == target_user:
            continue

        for movie, rating in users[neighbour].items():

            if movie in seen or movie in bad:
                continue

            score_sum[movie] = score_sum.get(movie, 0) + rating
            score_count[movie] = score_count.get(movie, 0) + 1

    predicted = []
    for movie, total in score_sum.items():
        avg = total / score_count[movie]
        predicted.append((movie, avg))

    predicted.sort(key=lambda x: x[1], reverse=True)
    return predicted[:top_k]


def anti_recommend(users, disliked, movies, clusters, labels, user_list, target_user, top_k=5):
    """
    Generuje antyrekomendacje (filmy, których użytkownik raczej nie polubi),
    na podstawie klastrów i średnich ocen innych użytkowników.

    Args:
        users (dict): {użytkownik: {film: ocena}}
        disliked (dict): {użytkownik: set(filmy)}
        movies (list): lista wszystkich filmów
        clusters (dict): {klaster: [użytkownicy]}
        labels (np.ndarray): etykiety klastrów KMeans
        user_list (list): lista użytkowników
        target_user (str): użytkownik docelowy
        top_k (int): ile filmów zwrócić

    Returns:
        list: [(film, średnia_ocena), ...] posortowane rosnąco
    """
    if target_user not in users:
        return []

    idx = user_list.index(target_user)
    user_cluster = labels[idx]
    neighbours = clusters.get(user_cluster, [])

    seen = set(users[target_user].keys())
    bad = disliked.get(target_user, set())

    score_sum = {}
    score_count = {}

    for neighbour in neighbours:
        if neighbour == target_user:
            continue

        for movie, rating in users[neighbour].items():

            if movie in seen or movie in bad:
                continue

            score_sum[movie] = score_sum.get(movie, 0) + rating
            score_count[movie] = score_count.get(movie, 0) + 1

    predicted = []
    for movie, total in score_sum.items():
        avg = total / score_count[movie]
        predicted.append((movie, avg))

    predicted.sort(key=lambda x: x[1])   # najmniejsza średnia = najgorsze
    return predicted[:top_k]


API_KEY = "90d0e92c"


def get_movie_info(movie):
    """
    Wyciągnięcie dodatkowych informacji dot. filmu
    :param movie: Tytuł filmu
    :return: Lista dodatkowych informacji o filmie
    """
    url = "http://www.omdbapi.com/"
    params = {"t": movie, "apikey": API_KEY}
    response = requests.get(url, params=params)

    if response.status_code != 200:
        print("Błąd połączenia", response.text)
        return {"Title": "None", "Year": "None", "Genre": "None"}

    data = response.json()
    if data.get("Response") == "False":
        return {"Title": "None", "Year": "None", "Genre": "None"}
    return data


def main():
    csv_path = "filmy.csv"

    # wczytanie danych
    users, movies = load_csv(csv_path)
    user_list = list(users.keys())

    print("Użytkownicy:")
    for i, u in enumerate(user_list):
        print(f"{i+1}. {u}")

    idx = int(input("\nWybierz użytkownika (numer): ")) - 1
    target_user = user_list[idx]

    # wyznaczanie disliked
    disliked = build_disliked(users)

    # budowanie rating_matrix
    rating_matrix = build_rating_matrix(users, movies, user_list)

    # wyliczenie optymalnego K
    best_k, inertias = find_optimal_k(rating_matrix)
    print(f"\nOptymalna liczba klastrów (Elbow Method): {best_k}")

    # klastrowanie z optymalnym K
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(rating_matrix)

    # tworzenie mapy klastrów
    clusters = {}
    for u, lbl in zip(user_list, labels):
        clusters.setdefault(lbl, []).append(u)

    # rekomendacje
    recs = recommend(users, disliked, movies, clusters, labels, user_list, target_user)
    print("\nTOP 5 REKOMENDACJI:")
    for movie, score in recs:
        info = get_movie_info(movie)

        print(f"  {movie}, rok produkcji: {info['Year']}, gatunek: {info['Genre']}")

    # antyrekomendacje
    bad = anti_recommend(users, disliked, movies, clusters, labels, user_list, target_user)
    print("\nTOP 5 ANTYREKOMENDACJI:")
    for movie, score in bad:
        info = get_movie_info(movie)
        print(f"  {movie}, rok produkcji: {info['Year']}, gatunek: {info['Genre']}")


if __name__ == "__main__":
    main()
