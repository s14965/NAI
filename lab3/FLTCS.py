






class pokoj:
    def __init__(self, grze):
        self.grzejnik = grze
    temperatura = 0
    statusOkna = "Zamknięte"
    osobyWPomieszczeniu = []
    
    def otworzOkno(self):
        self.statusOkna = "Otwarte"
        

class grzejnik:
    def __init__(self, maxTemp, nag, och):
        self.maksymalnaTemperatura = maxTemp
        self.tempoNagrzewania = nag
        self.tempoOchladzania = och
        
    temperatura = 0    
    
g1 = grzejnik(60, 5.5, 2)
g2 = grzejnik(65, 4.5, 3)
g3 = grzejnik(80, 8, 4.5)

p1 = pokoj(g1)

print(p1.statusOkna)
p1.otworzOkno()
print(p1.statusOkna)


