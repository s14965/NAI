# -*- coding: utf-8 -*-
import json
import compute_scores as cs

ratings_file = 'ratingsTest.json'
    
with open(ratings_file,'r') as f:
    data = json.loads(f.read())

users = list(data.keys())
users.pop(0)
eucScore_board = {}
pearScore_board = {}
finScore_board = {}

Pawel_Czapiewski_Movie_List = list(data['Pawel Czapiewski'].keys())
recommended = []
discouraged = []

for user in users:
    if user != "Pawel Czapiewski":
        eucScore_board.update({user:cs.euclidean_score(data,"Pawel Czapiewski",user)})
        pearScore_board.update({user:cs.pearson_score(data,"Pawel Czapiewski",user)})
        finScore_board.update({user:float(eucScore_board[user])*float(pearScore_board[user])})

#choosing best person and movies
personWithBestScore = ""
bestScore = -10
for person in users:
    if finScore_board[person] > bestScore:
        personWithBestScore = person
        bestScore = finScore_board[person]

personWithBestScore_MovieList = list(data[personWithBestScore].keys())
for i in range(len(personWithBestScore_MovieList)):
    if personWithBestScore_MovieList[i] not in Pawel_Czapiewski_Movie_List:
        recommended.append(personWithBestScore_MovieList[i])
        if len(recommended) == 5:
            break
print("Person with best score:", personWithBestScore)
print("Recommended:", recommended)


#choosing worst person and movies
personWithWorstScore = ""
worstScore = 10
for person in users:
    if finScore_board[person] < worstScore:
        personWithWorstScore = person
        worstScore = finScore_board[person]

personWithWorstScore_MovieList = list(data[personWithWorstScore].keys())
for i in range(len(personWithWorstScore_MovieList)):
    if personWithWorstScore_MovieList[i] not in Pawel_Czapiewski_Movie_List:
        discouraged.append(personWithWorstScore_MovieList[i])
        if len(discouraged) == 5:
            break
print("Person with worst score:", personWithWorstScore)
print("Discouraged:", discouraged)