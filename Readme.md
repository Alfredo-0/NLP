# Análisis Semántico

Top2Vec es un algoritmo para el modelamiento de topicos y búsqueda semántica. Este automaticamente detecta topicos presentes en el texto permite ver conceptos cercanos.


```python
import pandas as pd
import re
from top2vec import Top2Vec
```

    /usr/lib/python3/dist-packages/scipy/__init__.py:146: UserWarning: A NumPy version >=1.17.3 and <1.25.0 is required for this version of SciPy (detected version 1.26.3
      warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"
    2024-04-03 22:32:59.601043: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-04-03 22:33:05.815115: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT



```python
#print(Top2Vec.__doc__)
```

Los documentos se encuentran en formato .txt y separados por parrafos. En total hay 1310 párrafos.


```python
def documentos(input):
    with open(input, 'r') as file:
        content = file.read()
    return content.split("\n")

docs = documentos("documentos.txt") 
len(docs)
```




    1310



Ajuste de parámetros para la creación del modelo.


```python
umap_args = {'n_neighbors': 7,
             'n_components': 3,
             'metric': 'cosine',
             "random_state": 15}

hdbscan_args = {'min_cluster_size': 7,
                'min_samples':3,
                'metric': 'euclidean',
                'cluster_selection_method': 'eom'}
```


```python
model = Top2Vec(docs, 
                speed = "deep-learn", #fast-learn, deep-learn
                min_count = 4, 
                ngram_vocab = True,
                workers = 8,
                umap_args = umap_args,
                hdbscan_args = hdbscan_args,
                #embedding_model = "universal-sentence-encoder-multilingual-large",
                #split_documents = True,
               )
```

    2024-04-03 22:45:15,621 - top2vec - INFO - Pre-processing documents for training
    INFO:top2vec:Pre-processing documents for training
    2024-04-03 22:45:16,722 - top2vec - INFO - Downloading universal-sentence-encoder-multilingual model
    INFO:top2vec:Downloading universal-sentence-encoder-multilingual model
    2024-04-03 22:45:20,389 - top2vec - INFO - Creating joint document/word embedding
    INFO:top2vec:Creating joint document/word embedding
    2024-04-03 22:45:48,277 - top2vec - INFO - Creating lower dimension embedding of documents
    INFO:top2vec:Creating lower dimension embedding of documents
    2024-04-03 22:45:56,905 - top2vec - INFO - Finding dense areas of documents
    INFO:top2vec:Finding dense areas of documents
    2024-04-03 22:45:56,990 - top2vec - INFO - Finding topics
    INFO:top2vec:Finding topics


Se lograron extraer 48 topicos distintos y un vocabulario de 1-gramas y 2-gramas de 3277 palabras.


```python
print(len(model.vocab), model.get_num_topics())
```

    3277 48



```python
bigrams = []
for word in model.vocab:
    if len(word.split()) == 2:
        bigrams.append(word)
print(len(bigrams))
bigrams[10:20]
```

    587





    ['las formas',
     'el mercado',
     'las relaciones',
     'economia popular',
     'derechos laborales',
     'sin patron',
     'los ultimos',
     'nuevas formas',
     'la economia',
     'en numeros']



Podemos tambien ver los algunos de los tópicos y las palabras relevantes/relacionadas/cercanas.


```python
topic_words, word_scores, topic_nums = model.get_topics(num_topics = 10)
print(topic_words, topic_nums)
```

    [['de' 'pues' 'le' 'del' 'the' 'la' 'da' 'an' 'un' 'al' 'van' 'tal' 'of'
      'do' 'dos' 'les' 'given' 'respecto' 'it' 'as' 'el' 'that' 'que' 'se'
      'its' 'tion' 'on' 'los' 'so' 'at' 'una' 'tra' 'por' 'alla' 'para' 'ha'
      'ello' 'su' 'namely' 'thus' 've' 'is' 'to' 'en' 'per' 'decir' 'ii'
      'hacia' 'he' 'po']
     ['employment level' 'ingreso promedio' 'empleados' 'unions' 'equation'
      'salarios' 'trabajadores' 'salarial' 'bajos ingresos' 'wages'
      'empresarial capitalista' 'empleo' 'economics' 'economistas'
      'economia capitalista' 'los trabajadores' 'incumbent workers'
      'employment' 'empleos' 'socioeconomic' 'del ingreso'
      'baja productividad' 'remuneraciones' 'empleabilidad' 'workers are'
      'labour force' 'remuneracion' 'ingreso' 'empresas capitalistas'
      'salario' 'les trabajadores' 'marginal' 'worker' 'per worker' 'income'
      'produccion distribucion' 'workers' 'capitalist firm' 'trabajador'
      'inequality' 'socioeconomicas' 'trabajadores cuenta'
      'trabajadores familiares' 'workforce' 'wage' 'curve' 'socio economic'
      'be lower' 'clase trabajadora' 'masa marginal']
     ['economia capitalista' 'empresarial capitalista' 'informal economy'
      'economics' 'la economia' 'del capitalismo' 'el capitalismo'
      'capitalist totality' 'empresas capitalistas' 'capitalistas'
      'economia publica' 'capitalista tradicional' 'capitalismo'
      'economia informal' 'capitalist firm' 'capitalista' 'economia'
      'economias' 'economia po' 'capitalist' 'las economias'
      'economia empresarial' 'economy' 'popular economy' 'economic'
      'economistas' 'economies' 'capitalism' 'capitalist system'
      'economic activity' 'neoliberalismo' 'popular economies'
      'socio economic' 'desempleo' 'solidarity economy' 'economia popular'
      'extra economic' 'economists' 'economic discipline'
      'economic activities' 'economia urbana' 'socioeconomic'
      'produccion distribucion' 'employment level' 'economicamente'
      'neoliberal policies' 'labour force' 'economia mixta' 'empleo'
      'los trabajadores']
     ['economia capitalista' 'economia popular' 'informal economy'
      'popular economy' 'socio economic' 'economia po' 'popular economies'
      'economics' 'economia' 'economia informal' 'la economia'
      'economia empresarial' 'socioeconomic' 'economia publica' 'economic'
      'economy' 'economias' 'solidarity economy' 'economistas'
      'economic discipline' 'las economias' 'economists' 'economic activity'
      'socioeconomicas' 'economic initiatives' 'economic practices'
      'economies' 'economicas populares' 'economic activities'
      'economia mixta' 'relaciones economicas' 'economicamente' 'economica'
      'extra economic' 'capitalista tradicional' 'economia urbana'
      'economicas' 'economicamente activa' 'unidades economicas'
      'actividad economica' 'sistema economico' 'empresarial capitalista'
      'excedente economico' 'economico' 'economicos' 'del capitalismo'
      'capitalismo' 'el capitalismo' 'organizacion economica'
      'capitalist totality']
     ['economia capitalista' 'empresarial capitalista' 'solidarity economy'
      'capitalist totality' 'integracion social' 'social solidaria'
      'capitalista tradicional' 'socioeconomic' 'empresas capitalistas'
      'capitalist system' 'capitalista' 'capitalistas' 'economia publica'
      'socio economic' 'del capitalismo' 'capitalismo' 'el capitalismo'
      'capitalist firm' 'informal economy' 'socioeconomicas' 'capitalist'
      'ciencias sociales' 'politicas sociales' 'neoliberal policies'
      'economics' 'la economia' 'capitalism' 'economia informal'
      'neoliberalismo' 'economia' 'liberalization' 'economia empresarial'
      'economia po' 'sociales' 'una sociedad' 'popular economy'
      'estructura social' 'economicamente' 'economia mixta'
      'sistema economico' 'economy' 'economic' 'economias' 'economia popular'
      'las economias' 'social movements' 'desempleo' 'relaciones sociales'
      'economia urbana' 'la sociedad']
     ['socioeconomic' 'economic initiatives' 'socio economic'
      'socioeconomicas' 'economics' 'economists' 'economic practices'
      'relaciones economicas' 'economic discipline' 'organizacion economica'
      'economistas' 'economic activity' 'popular economy' 'economic'
      'organizaciones economicas' 'economia publica' 'las economias'
      'informal economy' 'popular economies' 'economy' 'economic activities'
      'solidarity economy' 'economia empresarial' 'economias' 'economies'
      'sistema economico' 'la economia' 'economia popular'
      'ciencias sociales' 'politicas sociales' 'economicas'
      'relaciones sociales' 'economia capitalista' 'las sociedades'
      'collective initiatives' 'societies' 'economicamente'
      'economicamente activa' 'economia informal' 'crecimiento economico'
      'asociativas' 'actividad economica' 'economia' 'actividades economicas'
      'asociaciones' 'community based' 'analytical' 'sociedades'
      'analytical framework' 'organizativas']
     ['collective initiatives' 'organizacion economica' 'organizativas'
      'organizaciones economicas' 'las organizaciones' 'organizacion'
      'social movements' 'political participation' 'organizational'
      'organizan' 'organization' 'associative' 'organizaciones'
      'those organisations' 'territorial organizations' 'organisations'
      'eps organisations' 'collective action' 'organisational'
      'organizations' 'these organizations' 'these organisations'
      'associations' 'economic initiatives' 'asociativas' 'institucionales'
      'the organisation' 'organisation' 'initiatives' 'asociaciones'
      'politicas sociales' 'militantes' 'iniciativas' 'las iniciativas'
      'institucional' 'asociacion' 'empresas capitalistas'
      'integracion social' 'movement' 'asociativa' 'institutional'
      'activity based' 'initiative' 'solidarity economy' 'these initiatives'
      'economic activity' 'activos' 'social solidaria' 'popular solidaria'
      'economicamente activa']
     ['economia capitalista' 'solidarity economy' 'socio economic'
      'socioeconomic' 'economia publica' 'empresarial capitalista'
      'socioeconomicas' 'informal economy' 'la economia' 'economia'
      'economia informal' 'economia mixta' 'economics' 'economic'
      'economia empresarial' 'economic activity' 'empresas capitalistas'
      'economic initiatives' 'integracion social' 'economias'
      'capitalist system' 'economia po' 'relaciones economicas'
      'capitalistas' 'capitalista' 'social solidaria' 'economicamente'
      'capitalista tradicional' 'economic activities' 'economicamente activa'
      'las economias' 'economy' 'capitalist totality' 'sistema economico'
      'capitalist firm' 'capitalist' 'del capitalismo' 'capitalismo'
      'economies' 'el capitalismo' 'popular economy' 'economia popular'
      'economic discipline' 'ciencias sociales' 'capitalism'
      'popular economies' 'organizacion economica' 'asociacion'
      'economic practices' 'relaciones sociales']
     ['popular economy' 'economia popular' 'popular economies'
      'economia publica' 'economicas populares' 'economistas'
      'economia urbana' 'la economia' 'economia po' 'ingreso promedio'
      'informal economy' 'economia' 'economias' 'population' 'economy'
      'ingreso total' 'las economias' 'economia informal' 'economists'
      'economics' 'empleos' 'estadisticas' 'employment level' 'economic'
      'economies' 'la poblacion' 'economia capitalista' 'socio economic'
      'empleo' 'economia empresarial' 'estos trabajadores' 'economia mixta'
      'socioeconomic' 'economicamente' 'los trabajadores' 'les trabajadores'
      'ciencias sociales' 'extra economic' 'salarios' 'socioeconomicas'
      'income' 'poblacion' 'employment' 'sectores populares' 'los ingresos'
      'desempleo' 'bajos ingresos' 'trabajadores cuenta' 'solidarity economy'
      'income per']
     ['les trabajadores' 'los trabajadores' 'trabajadores cuenta'
      'trabajadores' 'trabajadoras' 'workers' 'workers are' 'empleos'
      'estos trabajadores' 'horas trabajadas' 'incumbent workers'
      'trabajador' 'per worker' 'empleados' 'trabajadora' 'clase trabajadora'
      'employment level' 'salarios' 'trabajo asalariado'
      'trabajadores familiares' 'empleo' 'laborales' 'ocupacionales'
      'categoria ocupacional' 'are employed' 'employment' 'ingreso horario'
      'ocupaciones' 'ocupacional' 'laboral' 'jobs' 'salario' 'trabajan'
      'informal employment' 'worker' 'labores' 'del trabajo' 'empleabilidad'
      'workforce' 'que trabajan' 'insercion ocupacional' 'wages' 'desempleo'
      'ingreso promedio' 'salarial' 'trabajos' 'trabajar mas'
      'remuneraciones' 'economistas' 'ocupacion']] [0 1 2 3 4 5 6 7 8 9]


Procedemos a generar nubes de palabras para identificar las palabras clave de cada tópico. 


```python
for topic in [1,2, 3, 4, 5, 6, 7, 8, 9, 10]:
    model.generate_topic_wordcloud(topic)
```


    
![png](/Img/output_5_0.png)
    



    
![png](/Img/output_5_1.png)
    



    
![png](/Img/output_5_2.png)
    



    
![png](/Img/output_5_3.png)
    



    
![png](/Img/output_5_4.png)
    



    
![png](/Img/output_5_5.png)
    



    
![png](/Img/output_5_6.png)
    



    
![png](/Img/output_5_7.png)
    



    
![png](/Img/output_5_8.png)
    



    
![png](/Img/output_5_9.png)
    


Adicionalmente podemos generar nubes de palabras relacionadas con palabras identificadas en el vocabulario. En este caso particular podemos buscar los 5 tópicos mas relevantes/cercanos a las palabras "solidaridad y vida".


```python
topic_words, word_scores, topic_scores, topic_nums = model.search_topics(keywords=["solidaridad", "vida"], num_topics=6)
for topic in topic_nums:
    if topic != 0:
        model.generate_topic_wordcloud(topic)
```


    
![png](/Img//Img/output_7_0.png)
    



    
![png](/Img/output_7_1.png)
    



    
![png](/Img/output_7_2.png)
    



    
![png](/Img/output_7_3.png)
    



    
![png](/Img/output_7_4.png)
    

