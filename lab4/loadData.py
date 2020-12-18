# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 00:04:34 2020

@author: Nomys
"""
import openpyxl
import json

wb = openpyxl.load_workbook('data.xlsx')
ws = wb.active

class User:
    def __init__(self, row):
        self.movies = dict()
        self.name = str(row[0])
        self.fillMovieList(row)
        self.sortMovieListByRating_Descending_AndThenAlphabetically()
    
    def fillMovieList(self, row):
        for cell in range(1, len(row), 2):
            if row[cell] != None:
                self.movies.update({str(row[cell]):row[cell + 1]})
    
    def sortMovieListByRating_Descending_AndThenAlphabetically(self):
        tens = list()
        nines = list()
        eights = list()
        sevens = list()
        sixes = list()
        fives = list()
        fourths = list()
        threes = list()
        twos = list()
        ones = list() 
        for movie in self.movies:
            if self.movies[movie] == 10:
                tens.append(movie)
                tens.sort()
            elif self.movies[movie] == 9:
                nines.append(str(movie))
                nines.sort()
            elif self.movies[movie] == 8:
                eights.append(str(movie))
                eights.sort()
            elif self.movies[movie] == 7:
                sevens.append(str(movie))
                sevens.sort()
            elif self.movies[movie] == 6:
                sixes.append(str(movie))
                sixes.sort()
            elif self.movies[movie] == 5:
                fives.append(str(movie))
                fives.sort()
            elif self.movies[movie] == 4:
                fourths.append(str(movie))
                fourths.sort()
            elif self.movies[movie] == 3:
                threes.append(str(movie))
                threes.sort()
            elif self.movies[movie] == 2:
                twos.append(str(movie))
                twos.sort()
            elif self.movies[movie] == 1:
                ones.append(str(movie))
                ones.sort()
            
        self.movies.clear()
        for movie in tens:
            self.movies.update({str(movie):10})
        for movie  in nines:
            self.movies.update({str(movie):9})
        for movie in eights:
            self.movies.update({str(movie):8})
        for movie in sevens:
            self.movies.update({str(movie):7})
        for movie in sixes:
            self.movies.update({str(movie):6})
        for movie in fives:
            self.movies.update({str(movie):5})
        for movie in fourths:
            self.movies.update({str(movie):4})
        for movie in threes:
            self.movies.update({str(movie):3})
        for movie in twos:
            self.movies.update({str(movie):2})
        for movie in ones:
            self.movies.update({str(movie):1})
         
            
    def returnMovies(self):
        return self.movies
    
    #for debuging
    # def printMovies(self):
    #     print("===============")
    #     print(self.name, "Movie list \n ====")
    #     for movie in self.movies:
    #         print(movie, self.movies[str(movie)])
            
    
            
users = list()
AllMovies = list()

for row in ws.values:
    users.append(User(row))
    
    
data = {}
for person in users:
    data.update({person.name:person.returnMovies()})
    
with open('ratingsTest.json', 'w') as outfile:
    json.dump(data, outfile)