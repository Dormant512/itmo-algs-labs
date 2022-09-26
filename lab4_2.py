import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

cities=pd.read_table('coords.txt',sep='\t',header=None)
cities.columns=['x']
cities['y']=None
np.random.seed(2) #Fix random parameters to reproduce results on different machines

#distance calculation between coordinates
def CalDistance(x, y):
    return math.sqrt(x**2 + y**2)

def CalLength(cities, paths, start, end):
    length=0
    n=1
    for i in range(len(paths)):
        if i==0:
            length += CalDistance(start[0] - cities['x'][paths[i]], start[1] - cities['y'][paths[i]])
            n=n+1
        elif n < len(paths):
            length = length + CalDistance(cities['x'][paths[i]] - cities['x'][paths[i+1]], cities['y'][paths[i]] - cities['y'][paths[i+1]])
            n=n+1
        else:
            length = length + CalDistance(cities['x'][paths[i]] - end[0], cities['y'][paths[i]] - end[1])
    return length


for i in range(len(cities)):
    coordinate=cities['x'][i].split()
    cities['x'][i]=float(coordinate[0])
    cities['y'][i]=float(coordinate[1])

start=list(cities.iloc[0])
end=list(cities.iloc[0])
cities=cities.drop([0])
cities.index=[i for i in range(len(cities))]

#initiate path
paths=[i for i in range(len(cities))]

distance_1=0
distance_2=0
dif=0

for i in range(10):  
    newPaths_1 = list(np.random.permutation(paths))
    newPaths_2 = list(np.random.permutation(paths))
    distance_1 = CalLength(cities,newPaths_1,start,end)
    distance_2 = CalLength(cities,newPaths_2,start,end)
    dif_new = abs(distance_1-distance_2)
    if dif_new >= dif:
        dif = dif_new

#initiate accept possibility
Pr=0.5 

#initiate terperature
temp_0=dif/Pr
temp=temp_0
temp_min=temp/50

#iterations of internal circulation
k=10*len(paths) 

initialPath=paths.copy()
length=CalLength(cities,initialPath,start,end)
print("Path length on the first iteration:", length)

cities['order']=initialPath
cities_order=cities.sort_values(by=['order'])
plt.plot(cities_order['x'], cities_order['y'], 'bo', label="Coordinates") 
plt.show() 
plt.plot(cities_order['x'], cities_order['y'], 'bo', label="Coordinates")   
plt.plot(cities_order['x'],cities_order['y'], ls='--', label="First iteration")    
plt.show()   
 
#iteration`s counter 
counter=0 
initial_Path=list(np.random.permutation(paths))
length=CalLength(cities,initial_Path, start, end)
optimal_Path = initial_Path.copy()
optimal_Length=length
while temp>temp_min:
    for i in range(k):
        new_Paths=optimal_Path.copy()
        for j in range(int(temp_0/500)):
            a=0
            b=0
            while a==b:
                a=np.random.randint(0,len(paths))
                b=np.random.randint(0,len(paths))
            te=new_Paths[a]
            new_Paths[a]=new_Paths[b]
            new_Paths[b]=te
        new_Length=CalLength(cities, new_Paths, start, end)
        if new_Length<optimal_Length:
            optimal_Length=new_Length
            optimal_Path=new_Paths
        else:
             p=math.exp(-(new_Length-optimal_Length)/temp)
             r=np.random.uniform(low=0,high=1)
             if r<p:
                 optimal_Length=new_Length
                 optimal_Path=new_Paths
    back=np.random.uniform(low=0,high=1)
    if back>=0.85:
        temp=temp*2
    counter+=1 
    temp=temp_0/(1+counter)
    
print("Optimal path length:", optimal_Length)

#data update for optimal path plot
cities['order']=initial_Path
cities_order=cities.sort_values(by=['order'])

plt.plot(cities_order['x'], cities_order['y'], 'bo', label="Coordinates") 
plt.plot(cities_order['x'], cities_order['y'], ls='--', label="Last iteration")        
plt.show()    
