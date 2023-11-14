import dataPreperation as dp
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors

df2 = dp.df2
features = dp.features

min_max_scaler = MinMaxScaler()
features = min_max_scaler.fit_transform(features)


model = neighbors.NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
model.fit(features)
dist, idlist = model.kneighbors(features)


def BookRecommender(book_name):
    book_list_name = []
    book_id = df2[df2['title'] == book_name].index
    if book_id.size == 0 :
        return "Books not found in Database ! please try again with detailed name in format \"Harry Potter and the Half-Blood Prince (Harry Potter  #6)\"  !"
    book_id = book_id[0]
    for newid in idlist[book_id]:
        book_list_name.append(df2.loc[newid].title)
    return book_list_name

if __name__ == "__main__" :
    
    book_name = input("Name of Book you liked and want similar recommendation of  \nName should in format : \"The Fellowship of the Ring (The Lord of the Rings  #1)\" ")

    BookNames = BookRecommender(book_name)

    for i in range(1,len(BookNames)) :
        print(f"{i} : {BookNames[i]}")