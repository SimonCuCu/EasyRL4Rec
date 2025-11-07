import pandas as pd, pathlib
root = pathlib.Path("data/MovieLens/data_raw/ml-100k")
out = root.parent  # data_raw/

# ratings.dat → UserID::MovieID::Rating::Timestamp
cols = ["user_id","item_id","rating","timestamp"]
ratings = pd.read_csv(root/"u.data", sep="\t", names=cols)
def write_double_colon(df, path):
    path = pathlib.Path(path)
    with path.open("w", encoding="utf-8") as f:
        for row in df.itertuples(index=False):
            f.write("::".join(map(str, row)) + "\n")
write_double_colon(ratings, out/"ratings.dat")

# users.dat → UserID::Gender::Age::Occupation::Zip-code
users = pd.read_csv(root/"u.user", sep="|",
                    names=["user_id","age","gender","occupation","zip_code"])
users["gender"] = users["gender"].str.upper()
users["zip_code"] = users["zip_code"].str.split("-").str[0]
occupation_order = pd.read_csv(root/"u.occupation", header=None, names=["occupation"])["occupation"].tolist()
occ_to_id = {occ: idx for idx, occ in enumerate(occupation_order)}
users["occupation"] = users["occupation"].map(occ_to_id).astype(int)
users = users[["user_id","gender","age","occupation","zip_code"]]
write_double_colon(users, out/"users.dat")

# movies.dat → MovieID::Title (with year)::Genres
movies = pd.read_csv(root/"u.item", sep="|", header=None, encoding="latin1",
                     names=["item_id","title","release","video","imdb",
                            *[f"g{i}" for i in range(19)]])
genre_names = pd.read_csv(root/"u.genre", sep="|", header=None,
                          names=["genre","id"]).dropna().sort_values("id")["genre"].tolist()
def collect_genres(row):
    active = [genre_names[i] for i in range(len(genre_names)) if row[f"g{i}"]==1]
    return "|".join(active) if active else "(no genres listed)"
movies["genres"] = movies.apply(collect_genres, axis=1)
def extract_year(release):
    if isinstance(release, str) and release[-4:].isdigit():
        return release[-4:]
    return "0000"
movies["year"] = movies["release"].apply(extract_year)
movies["movie_title"] = movies.apply(lambda row: f"{row['title']} ({row['year']})", axis=1)
movies_out = movies[["item_id","movie_title","genres"]].rename(columns={"movie_title":"title","genres":"genre"})
write_double_colon(movies_out, out/"movies.dat")
