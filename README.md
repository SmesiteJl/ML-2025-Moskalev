# Машинное обучение



## Оглавление

1. [K-Means на датасете «Ирисы Фишера»](#1-k-means-на-датасете-ирисы-фишера)  
2. [Предобработка и линейная регрессия (Housing Dataset)](#2-предобработка-и-линейная-регрессия-housing-dataset)  
3. [Распознавание лица, жестов и эмоций в реальном времени](#3-распознавание-лица-жестов-и-эмоций-в-реальном-времени)  
4. [Генетический алгоритм маршрута (с приоритетами и картой)](#4-генетический-алгоритм-маршрута-с-приоритетами-и-картой)  

---

## 1. K-Means на датасете «Ирисы Фишера»

**Часть 1:**  
- Используется `sklearn` для определения оптимального числа кластеров методами локтя и силуэт-метрики.  
- Автоматический вывод оптимального количества кластеров.  
- Визуализация локтевой диаграммы и силуэт-анализа для обоснования выбора.  

**Часть 2:**  
- Самописная реализация алгоритма K-Means.  
- Пошаговая визуализация каждого шага: сдвиг центроидов, перераспределение точек.  
- Цветовое выделение кластеров для удобства восприятия.  

**Часть 3:**  
- Итоговая визуализация во всех 2D-проекциях (все пары признаков).  

**Пример результата:**  

Оптимальное количество кластеров: 3


**Зависимости:**  
```bash
pip install pandas matplotlib seaborn numpy scikit-learn
````

---

## 2. Предобработка и линейная регрессия (Housing Dataset)

**Основные задачи:**

* Загрузка и очистка данных, включая удаление высококоррелирующих признаков для предотвращения мультиколлинеарности.
* Нормализация признаков.
* Снижение размерности с помощью PCA до 2 главных компонент для визуализации 3D-графика зависимости цены (`SalePrice`).
* Разделение данных на обучающую и тестовую выборки.
* Обучение модели линейной регрессии, вычисление RMSE на тесте.
* Применение Lasso-регрессии с графиком зависимости RMSE от коэффициента регуляризации.
* Определение наиболее влияющего признака (например, `GrLivArea`).

**Зависимости:**

```bash
pip install pandas matplotlib seaborn scikit-learn
```

---

## 3. Распознавание лица, жестов и эмоций в реальном времени

**Описание:**

* С помощью веб-камеры определяется лицо пользователя.
* Считается количество поднятых пальцев для управления выводом:

  * 1 палец — отображается имя пользователя.
  * 2 пальца — отображается фамилия.
  * 3 пальца — запускается распознавание текущей эмоции.
* При появлении в кадре чужого лица выводится сообщение "неизвестный".

**Обработка ошибок:**

* Если пальцев нет или количество не соответствует 1-3, вывод информации не происходит.
* В случае нескольких лиц определяется только первое обнаруженное лицо.

**Используемые модели:**

* DeepFace для распознавания эмоций и идентификации лица.

**Зависимости:**

```bash
pip install opencv-python mediapipe deepface
```

---

## 4. Генетический алгоритм маршрута (с приоритетами и картой)

**Описание задачи:**

* Генетический алгоритм формирует маршрут по заданным точкам с учётом приоритетов (весов) и ограничения по времени (например, 50 минут).
* Начальная точка маршрута фиксирована.
* Фитнес-функция — сумма весов посещённых точек, маршрут выбирается с максимальным весом без превышения лимита времени.

**Визуализация:**

* Итоговый маршрут отображается на интерактивной карте с помощью `folium`.
* HTML-карта сохраняется в `route_map.html` и автоматически открывается после выполнения программы.

**Особенности:**

* Время движения рассчитывается на основе расстояния и средней скорости (например, 5 км/ч).
* Классический подход GA: отбор, скрещивание, мутация.

**Формат файла `points.csv`:**

```csv
name,lat,lon,weight
Place A,55.75,37.61,5
Place B,55.76,37.62,3
...
```

**Пример вывода:**

```
Лучший маршрут: ['Place A', 'Place C', 'Place D'] — суммарный приоритет: 17.2
```

**Зависимости:**

```bash
pip install folium geopy
```

---

