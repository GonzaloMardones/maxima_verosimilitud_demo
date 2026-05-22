# Máxima Verosimilitud aplicada a no-show de pasajeros

Ejemplo didáctico de **Máxima Verosimilitud (Maximum Likelihood Estimation, MLE)** usando un caso ficticio pero plausible de aviación: estimar la tasa de *no-show* de pasajeros a partir de datos sintéticos de vuelos y traducir esa estimación en una consecuencia operacional simplificada.

> **Nota importante**  
> Este proyecto no describe una política real de overbooking ni el funcionamiento de una aerolínea específica. Es un ejercicio educativo para explicar MLE, incertidumbre estadística y supuestos de modelamiento usando un dominio familiar: vuelos, reservas y pasajeros que no se presentan al embarque.

---

## Objetivo

El objetivo del proyecto es responder una pregunta estadística simple:

> Dado un conjunto de vuelos observados, ¿qué valor de `p` —la probabilidad de no-show— hace más probables los datos que vimos?

Para eso, el código:

1. Genera un dataset sintético de vuelos.
2. Simula reservas y no-shows por vuelo.
3. Estima la tasa de no-show usando MLE.
4. Calcula un intervalo de confianza aproximado al 95%.
5. Genera visualizaciones para explicar:
   - variabilidad vuelo a vuelo,
   - curva de log-verosimilitud,
   - convergencia del estimador,
   - análisis simplificado de overbooking,
   - ciclo conceptual de MLE.

---

## Caso simulado

El escenario considera:

- 180 vuelos ficticios.
- Entre 140 y 180 reservas por vuelo.
- Una tasa real usada en la simulación de `p = 0.08`.
- No-shows generados con una distribución Binomial.
- Estimación de `p` sin conocer directamente el valor usado en la simulación.

El modelo asumido es:

```math
k_i ~ Binomial(n_i, p)
```

donde:

- `n_i` es la cantidad de reservas del vuelo `i`.
- `k_i` es la cantidad de pasajeros que no se presentaron.
- `p` es la probabilidad de no-show que queremos estimar.

---

## Metodología

La función de verosimilitud para todos los vuelos es:

```math
L(p | datos) = producto_i C(n_i, k_i) p^{k_i}(1-p)^{n_i-k_i}
```

Para mayor estabilidad numérica se usa la log-verosimilitud:

```math
log L(p) = suma_i [k_i log(p) + (n_i-k_i) log(1-p)] + C
```

El estimador MLE para este caso binomial tiene una solución cerrada:

```math
p_hat_MLE = suma_i k_i / suma_i n_i
```

Además, se calcula un error estándar aproximado mediante información de Fisher:

```math
SE(p_hat) = sqrt(p_hat(1-p_hat) / suma_i n_i)
```

---

## Resultados principales

Con la simulación incluida en el script, el estimador recupera una tasa cercana al valor usado para generar los datos:

- `p_MLE ≈ 0.0788`
- Tasa estimada de no-show: **7.88%**
- Intervalo de confianza 95% aproximado: **[7.57%, 8.19%]**

Estos valores pueden variar si cambias la semilla aleatoria, la cantidad de vuelos o el rango de reservas por vuelo.

---

## Visualizaciones generadas

| Archivo | Descripción |
|---|---|
| `01_histograma_observado.png` | Distribución de la tasa de no-show observada por vuelo. |
| `02_log_verosimilitud.png` | Curva de log-verosimilitud y máximo en `p_MLE`. |
| `03_convergencia.png` | Evolución de la estimación acumulada y su intervalo de confianza al aumentar la cantidad de vuelos. |
| `04_overbooking.png` | Ejemplo hipotético de riesgo de denied boarding para distintos niveles de reservas vendidas. |
| `05_diagrama_ciclo.png` | Diagrama conceptual del ciclo de MLE: datos, modelo, verosimilitud, optimización y uso responsable. |

---

## Estructura sugerida del repositorio

```text
.
├── max_verosimilitud_demo.py
├── dataset_no_shows.csv
├── README.md
├── outputs/
│   ├── 01_histograma_observado.png
│   ├── 02_log_verosimilitud.png
│   ├── 03_convergencia.png
│   ├── 04_overbooking.png
│   └── 05_diagrama_ciclo.png
└── articulo_linkedin_v2.md
```

> Recomendación: en `max_verosimilitud_demo.py`, cambia la variable `OUT` para guardar los outputs en una carpeta local del repositorio, por ejemplo:
>
> ```python
> OUT = "./outputs"
> ```

---

## Instalación

```bash
git clone https://github.com/tu-usuario/tu-repo.git
cd tu-repo
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib scipy
```

En Windows:

```bash
.venv\Scripts\activate
pip install numpy pandas matplotlib scipy
```

---

## Ejecución

```bash
python max_verosimilitud_demo.py
```

El script imprimirá en consola una muestra del dataset, el total de reservas, el total de no-shows, la estimación MLE y el intervalo de confianza.

También generará el dataset sintético y las imágenes en la carpeta definida por la variable `OUT`.

---

## Descripción del código

El archivo `max_verosimilitud_demo.py` está organizado en cinco bloques principales:

### 1. Configuración visual

Define una paleta de colores y parámetros globales de Matplotlib para mantener consistencia visual en todos los gráficos.

### 2. Generación de datos sintéticos

Simula 180 vuelos ficticios. Para cada vuelo:

- genera un número aleatorio de reservas entre 140 y 180,
- simula la cantidad de no-shows usando una distribución Binomial,
- calcula la tasa de no-show observada por vuelo,
- guarda el resultado en `dataset_no_shows.csv`.

### 3. Estimación por Máxima Verosimilitud

Implementa:

- una función de log-verosimilitud,
- una función negativa para optimización numérica,
- una estimación numérica usando `scipy.optimize.minimize`,
- la solución `sum(no_shows) / sum(bookings)`,
- el error estándar y el intervalo de confianza al 95%.

### 4. Visualizaciones estadísticas

Genera gráficos para explicar visualmente:

- por qué la tasa observada varía vuelo a vuelo,
- cómo MLE encuentra el valor de `p` que maximiza la log-verosimilitud,
- cómo mejora la precisión a medida que aumentan los datos observados.

### 5. Traducción a decisión hipotética

Usa la tasa estimada para simular una consecuencia operacional simplificada: el riesgo de que se presenten más pasajeros que asientos disponibles bajo distintos niveles de reservas vendidas.
---

## Limitaciones del ejemplo

Este proyecto simplifica varios aspectos importantes:

1. **`p` se asume constante**  
   En la realidad, la probabilidad de no-show puede depender de ruta, día de la semana, temporada, tipo de tarifa, anticipación de compra o perfil de pasajero.

2. **Se asume independencia**  
   El modelo Binomial supone que los eventos son independientes. En la práctica, pueden existir correlaciones por clima, eventos, conexiones o disrupciones operacionales.

3. **No optimiza una política real**  
   El análisis de overbooking es una traducción hipotética del parámetro estimado. Una decisión real requeriría costos, restricciones regulatorias, restricciones operacionales y modelos más complejos.

---

## Posibles extensiones

- Modelar `p` con regresión logística.
- Incorporar variables por vuelo: día de semana, horario, ruta, tarifa o anticipación de compra.
- Comparar MLE binomial simple vs. modelos segmentados.
- Simular sensibilidad del resultado a distintos tamaños de muestra.
- Agregar una función de costo económico para denied boarding y asientos vacíos.
- Convertir el análisis en notebook interactivo.

---

## Tecnologías utilizadas

- Python
- NumPy
- Pandas
- SciPy
- Matplotlib

---

## Uso educativo

Este repositorio está pensado como material de apoyo para explicar:

- Máxima Verosimilitud.
- Distribución Binomial.
- Log-verosimilitud.
- Intervalos de confianza.
- Incertidumbre estadística.
- Comunicación de supuestos en modelos aplicados.

---

## Licencia

Puedes usar este proyecto con fines educativos y de divulgación. Si lo reutilizas, considera citar o enlazar el repositorio original.

