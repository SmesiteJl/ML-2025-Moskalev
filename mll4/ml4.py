import folium
from geopy.distance import geodesic
import random
import webbrowser
import csv

class Point:
    def __init__(self, name, lat, lon, weight):
        self.name = name
        self.lat = float(lat)
        self.lon = float(lon)
        self.weight = float(weight)

def load_points_from_csv(filepath):
    points = []
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            points.append(Point(row['name'], row['lat'], row['lon'], row['weight']))
    return points

points = load_points_from_csv('points.csv')

TIME_LIMIT = 50  
SPEED_KMH = 5 
START_POINT = points[0]
POP_SIZE = 50
GENS = 100

def time_between(p1, p2):
    distance_km = geodesic((p1.lat, p1.lon), (p2.lat, p2.lon)).km
    return (distance_km / SPEED_KMH) * 60 

def fitness(route):
    time = 0
    score = 0
    current = START_POINT
    for point in route:
        time += time_between(current, point)
        if time > TIME_LIMIT:
            return 0
        score += point.weight
        current = point
    return score

def generate_individual():
    return random.sample(points[1:], k=random.randint(2, len(points) - 1))

def crossover(r1, r2):
    min_len = min(len(r1), len(r2))
    if min_len < 2:
        return random.choice([r1, r2])
    cut = random.randint(1, min_len - 1)
    child = r1[:cut] + [p for p in r2 if p not in r1[:cut]]
    return child

def mutate(route):
    if len(route) > 1:
        i = random.randint(0, len(route) - 1)
        candidates = [p for p in points[1:] if p not in route]
        if candidates:
            route[i] = random.choice(candidates)
    return route

population = [generate_individual() for _ in range(POP_SIZE)]
for gen in range(GENS):
    population.sort(key=fitness, reverse=True)
    next_gen = population[:10] 
    while len(next_gen) < POP_SIZE:
        p1, p2 = random.sample(population[:25], 2)
        child = crossover(p1, p2)
        if random.random() < 0.3:
            child = mutate(child)
        next_gen.append(child)
    population = next_gen

best = max(population, key=fitness)
print("Самый крутой маршрут:", [p.name for p in best], "Сумма весов:", fitness(best))

map = folium.Map(location=[START_POINT.lat, START_POINT.lon], zoom_start=13)
folium.Marker([START_POINT.lat, START_POINT.lon], tooltip="Start", icon=folium.Icon(color="green")).add_to(map)

cur = START_POINT
for point in best:
    folium.Marker([point.lat, point.lon], tooltip=f"{point.name} (вес {point.weight})").add_to(map)
    folium.PolyLine([[cur.lat, cur.lon], [point.lat, point.lon]], color="blue").add_to(map)
    cur = point

map_file = "route_map.html"
map.save(map_file)
webbrowser.open(map_file)
