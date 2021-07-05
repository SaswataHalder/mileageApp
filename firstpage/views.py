from django.http import HttpResponse
from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn import preprocessing, pipeline, metrics, model_selection, ensemble
from sklearn_pandas import DataFrameMapper
from csv import writer

# Create your views here.
temp = {}


def index(request):
    temp['cylinders'] = 7
    temp['displacement'] = 400
    temp['horsepower'] = 200
    temp['weight'] = 2000
    temp['model year'] = 81
    temp['origin'] = 2
    context = {'temp': temp, 'model_year': temp['model year']}
    return render(request, 'home.html', context)


def predict(request):
    car_df = pd.read_csv('./mpg.csv')
    mapper = DataFrameMapper([
        (['cylinders', 'displacement', 'horsepower', 'weight', 'model year'], preprocessing.StandardScaler()),
        (['origin'], preprocessing.OneHotEncoder())
    ])
    pipeline_obj = pipeline.Pipeline([
        ('mapper', mapper),
        ("model", ensemble.RandomForestRegressor())
    ])
    X = ['cylinders', 'displacement', 'horsepower', 'weight', 'model year', 'origin']
    Y = ['mpg']
    pipeline_obj.fit(car_df[X], car_df[Y])

    if request.method == 'POST':
        temp['cylinders'] = int(request.POST.get('cyl'))
        temp['displacement'] = float(request.POST.get('dis'))
        temp['horsepower'] = float(request.POST.get('hp'))
        temp['weight'] = int(request.POST.get('w'))
        temp['model year'] = int(request.POST.get('model'))
        temp['origin'] = int(request.POST.get('origin'))
        print(temp['origin'])
        model_year = temp['model year']

    testData = pd.DataFrame({'x': temp}).transpose()

    scoreval = pipeline_obj.predict(testData)[0]

    context = {'scoreval': scoreval, 'temp': temp, 'model_year': model_year}
    return render(request, 'home.html', context)


def update(request):
    cyl = int(request.POST.get('cyl'))
    dis = float(request.POST.get('dis'))
    hp = float(request.POST.get('hp'))
    w = int(request.POST.get('w'))
    model = int(request.POST.get('model'))
    origin = int(request.POST.get('origin'))
    mpg = float(request.POST.get('mpg'))
    List=[mpg,cyl,dis,hp,w,model,origin]
    with open('mpg.csv', 'a') as f_object:
        writer_object = writer(f_object,delimiter=',',lineterminator='\n')
        writer_object.writerow(List)
        f_object.close()
    return render(request, 'thanks.html')
