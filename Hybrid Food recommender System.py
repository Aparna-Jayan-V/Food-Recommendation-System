
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn.metrics.pairwise import cosine_similarity
import random
import nltk


df=pd.read_csv('super.csv')

df['diet_name'] = df['diet_name'].replace('Non Vegetarian','NonVegetarian')

df['diet_name'] = df['diet_name'].replace('Gluten Free','GlutenFree')

df['diet_name']=df['diet_name'].replace('100% Halal','Halal')


h=df['diet_name']


# Content based recommendations

# For content-based recommendations, we will be looking at diet_name and searching for similar items.
# To find the similarity between the two items we will be using the cosine similarity metric.
# To measure the similarity between items, we need to present the genre line in a more formalized form.
# To do so, we will use a bag of words model. So for each item, we will get a vector of 21 values, 
# indicating which genres it belongs to.

vectorizer = CountVectorizer()
v = vectorizer.fit_transform(df['diet_name'].values)
feature_name = vectorizer.get_feature_names_out()


# Creating a dataframe for genre indicators and combining them in one list for each item.

genre_bow = pd.DataFrame(v.toarray(), columns=feature_name)
genre_bow['combined']= genre_bow.values.tolist()


# Replacing genres line with a bag of words representation.

df['diet_name'] = genre_bow['combined']


# The function below returns IDs for top N items similar to the given one.

def get_cossim(itemId, top):
    # Creating dataframe with only IDs and diet names
    items_to_search = df[['item_id', 'diet_name']]
    # Remove the ID of the items we are measuring distance to
    items_to_search = items_to_search[items_to_search.item_id != itemId]
    # Saving distances to new column
    items_to_search['dist'] = items_to_search['diet_name'].apply(lambda x: cosine_similarity(np.array(x).reshape(1, -1), np.array(df.loc[df['item_id'] == itemId]['diet_name'].values[0]).reshape(1, -1)))
    # Remove the diet column
    items_to_search = items_to_search.drop(columns=['diet_name'])
    # Distance value is in the list inside of the list so we need to unpack it
    items_to_search = items_to_search.explode('dist').explode('dist')
    # Sort the data and return top values
    return items_to_search.sort_values(by=['dist'], ascending=False)['item_id'].head(top).values


# The next function takes 10 top-rated items by a selected user and returns 5 similar items for each of those.

def get_similar(userId):
    # Take all the items watched by user
    items_ordered_by_user = df[df.user_id == userId]
    # Only 4.5 or higher rating filtered
    items_ordered_by_user = items_ordered_by_user[items_ordered_by_user['item_rating'] > 1]
    # Taking top 10 with highest ratings
    top_items_user = (items_ordered_by_user.sort_values(by="item_rating", ascending=False).head(10))
    top_items_user['ordered_item_id'] = top_items_user['item_id']
    top_items_user = top_items_user[['user_id', 'ordered_item_id']]
    # Find 5 similar items for each of the selected above
    top_items_user['similar'] = top_items_user['ordered_item_id'].apply(lambda x: (get_cossim(x, 5)))
    # Remove items that user have already ordered from recommendations
    result = [x for x in np.concatenate(top_items_user['similar'].values, axis=0).tolist() if x not in top_items_user.ordered_item_id.values.tolist()]
    return result


# 'get_top' function returns top N recommended items sorted by mean user rating. (only items with 1 or more ratings are used).

def get_top(id, top):     #userid
    # taking items that user may like
    smlr = get_similar(id)    
    # Calculating mean rating for every item
    ratings_mean_count = pd.DataFrame(df.groupby('item_id')['item_rating'].mean())
    ratings_mean_count['rating_counts'] = pd.DataFrame(df.groupby('item_id')['item_rating'].count())
    # Sorting items with 1 or more ratings by users
    ratings_mean_count = ratings_mean_count[ratings_mean_count['rating_counts'] > 1]
    # Returning top N items sorted by rating
    return ratings_mean_count[ratings_mean_count.index.isin(smlr)].sort_values(by=['item_rating'], ascending=False).head(top)


# Collaborative filtering

user_ids = df["user_id"].unique().tolist()
# Reassign user IDs
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}

item_ids = df["item_id"].unique().tolist()

# Reassign item IDs
item2item_encoded = {x: i for i, x in enumerate(item_ids)}
item_encoded2item = {i: x for i, x in enumerate(item_ids)}

df["user"] = df["user_id"].map(user2user_encoded)
df["item"] = df["item_id"].map(item2item_encoded)

num_users = len(user2user_encoded)
num_items = len(item_encoded2item)
df["item_rating"] = df["item_rating"].values.astype(np.float32)
# min and max ratings will be used to normalize the ratings later
min_rating = min(df["item_rating"])
max_rating = max(df["item_rating"])

print("Number of users: {}, Number of Items: {}, Min rating: {}, Max rating: {}".format(
        num_users, num_items, min_rating, max_rating))

# Normalizing ratings and splitting data.

df = df.sample(frac=1, random_state=42)
x = df[["user", "item"]].values
# Normalize the targets between 0 and 1. Makes it easy to train.
y = df["item_rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
# Assuming training on 90% of the data and validating on 10%.
train_indices = int(0.8 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:],)

# Defining model structure.

EMBEDDING_SIZE = 50

class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_items, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.item_embedding = layers.Embedding(
            num_items,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.item_bias = layers.Embedding(num_items, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        item_vector = self.item_embedding(inputs[:, 1])
        item_bias = self.item_bias(inputs[:, 1])
        dot_user_item = tf.tensordot(user_vector, item_vector, 2)
        # Add all the components (including bias)
        x = dot_user_item + user_bias + item_bias
        # The sigmoid activation forces the rating to between 0 and 1
        return tf.nn.sigmoid(x)

model = RecommenderNet(num_users, num_items, EMBEDDING_SIZE)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(lr=0.0005)
)


# Training the model in 15 epoches with batch size of 64 and learning rate of 0.0005.

history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=60,
    epochs=15,
    verbose=1,
    validation_data=(x_val, y_val),
)

# Results

# Set the ID of a user that we will be recommending items to.

userId = 'a397e455-41b3-4774-bff0-5eebe87405af'

# Get top 10 recommendations based on previously ordered items.
top=get_top(userId,10)

content_rec = top.index.values.tolist()

content_rec = top.index.values.tolist()


# Get the top 10 recommendations based on similar users' preferences.

# Searching for items that user already ordered
items_ordered_by_user = df[df.user_id == userId]

# Searching for items that user haven't ordered yet
items_not_ordered = df[
    ~df["item_id"].isin(items_ordered_by_user.item_id.values)
]["item_id"]

items_not_ordered = list(
    set(items_not_ordered).intersection(set(item2item_encoded.keys()))
)

items_not_ordered = [[item2item_encoded.get(x)] for x in items_not_ordered]
user_encoder = user2user_encoded.get(userId)
user_item_array = np.hstack(
    ([[user_encoder]] * len(items_not_ordered), items_not_ordered)
)
# Predicting ratings for items
ratings = model.predict(user_item_array).flatten()
# Sorting predicted ratings and taking top 10
top_ratings_indices = ratings.argsort()[-10:][::-1]
# Getting the actual IDs for items
recommended_item_ids = [
    item_encoded2item.get(items_not_ordered[x][0]) for x in top_ratings_indices
]

# Mixing all recommendations in one list and randomly taking 10 of them to recommend.

df['diet_name']=h

df4=pd.DataFrame(df[['item_id','item_name','diet_name','category_name']])

dff=df4.drop_duplicates()

# Mixing all recommendations in one list and randomly taking 10 of them to recommend.

print("Showing recommendations for user: {}".format(userId))
print("====" * 10)
print("Items with high ratings from user")
print("----" * 10)
top_items_user = (
    items_ordered_by_user.sort_values(by="item_rating", ascending=False)
    .head(20)
    .item_id.values
)
df_rows = dff[dff['item_id'].isin(top_items_user)]
for row in df_rows.itertuples():
    print(row.item_name, ":", row.category_name, ":",row.diet_name)
print("----" * 10)
print("Top 10 item recommendations")
print("----" * 10)
to_recommend = random.sample((content_rec + recommended_item_ids), 10)
recommended_items = dff[dff["item_id"].isin(to_recommend)]
for row in recommended_items.itertuples():
    print(row.item_name, ":", row.category_name,":",row.diet_name)  

import pickle

pickle.dump(content_rec, open('content_recommendation.pkl', 'wb'))
pickle.dump(recommended_item_ids, open('recommended_item_ids.pkl', 'wb'))
pickle.dump(dff, open('data.pkl', 'wb'))
pickle.dump(df, open('data_user.pkl', 'wb'))



