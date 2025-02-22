# %% [markdown]
# # Trabajo Final
# ### Integrantes Grupo 1:
# *   Juan Pablo Arrázola
# *   Paolo Brito
# *   José Gutierrez
# *   Horacio Molina

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# %% [markdown]
# # UPB POSGRADO ESTADÍSTICA APLICADA A CIENCIA DE DATOS TRABAJO FINAL Julio 2024 
# ### Tarea Estadística Inferencial

# %% [markdown]
# ### 1. Carga el dataset de tu número de grupo de la carpeta base de datos de Moodle 

# %%
df=pd.read_csv('RetailNuevo1.csv')

# %%
df.shape

# %%
df.columns

# %% [markdown]
# ### 2. Expresa las ventas segmentadas por estado, por modo de entrega (Ship Mode) y por subcategoría.

# %%
ventas_segmentadas = df.groupby(['State', 'Ship Mode', 'Sub-Category'])['Sales'].sum().reset_index()

ventas_segmentadas

# %% [markdown]
# ### 3. Por cada categoría haga un diagrama de torta que exprese el porcentaje de ventas de cada subcategoría.

# %%
df_agrupado_categoria = df.groupby(['Category', 'Sub-Category'])['Sales'].sum().reset_index()

categorias = df_agrupado_categoria['Category'].unique()

for categoria in categorias:
    df_filtrado = df_agrupado_categoria[df_agrupado_categoria['Category'] == categoria]

    plt.figure(figsize=(8,8))
    plt.pie(df_filtrado['Sales'], labels=df_filtrado['Sub-Category'], autopct='%1.1f%%')
    plt.title(f'Porcentaje de Ventas por Subcategoría en {categoria}')
    plt.show()

# %% [markdown]
# ### 4. Da un intervalo de confianza para la media de ventas de la tienda principal (la de mayor venta) al 90% y al 98% de confiabilidad. Si consideramos que se van a realizar 500 ventas en dicha tienda da una estimación de cuanto podemos ganar en dicha tienda a las mismas confiabilidades que hemos considerado en el punto anterior.

# %%
tienda_principal =df.groupby('City')['Sales'].sum().sort_values(ascending=False).idxmax()
df_tienda = df[df['City'] == tienda_principal]  

df_tienda.shape

# %%
n = len(df_tienda)
media = df_tienda['Sales'].mean()
desv_est = df_tienda['Sales'].std()
print(n,media,desv_est)

# %%
confianzas = [0.90, 0.98]
ventas_estimadas=500

# %%
for confianza in confianzas:
    alpha = 1 - confianza  
    t_critico = stats.t.ppf(1 - alpha/2, df=n-1) 
    margen_error = t_critico * (desv_est / np.sqrt(n))
    
    intervalo_inf = media - margen_error
    intervalo_sup = media + margen_error
    
    print(f"Intervalo de confianza al {int(confianza*100)}%: ({intervalo_inf:.2f}, {intervalo_sup:.2f})")

    ganancia_min = intervalo_inf * ventas_estimadas
    ganancia_max = intervalo_sup * ventas_estimadas
    
    print(f"Estimación de ganancias en {ventas_estimadas} ventas al {int(confianza*100)}%: ({ganancia_min:.2f}, {ganancia_max:.2f})\n")


# %% [markdown]
# ## 5. Verifica si estadísticamente los siguientes estados tienen las mismas ventas. 

# %% [markdown]
# *   (a) Washington y Arkansas. 
# *   (b) Arkansas y Maryland. 
# *   (c) Minnesota y Montana 
# *   (d) South Carolina y Connecticut

# %%
# pares_estados = [
#     ('Washington', 'Arkansas'),
#     ('Arkansas', 'Maryland'),
#     ('Minnesota', 'Montana'),
#     ('South Carolina', 'Connecticut')
# ]

# alpha = 0.05

# for estado1, estado2 in pares_estados:
#     ventas_estado1 = df[df['State'] == estado1]['Sales']
#     ventas_estado2 = df[df['State'] == estado2]['Sales']
    

#     t_stat, p_valor = stats.ttest_ind(ventas_estado1, ventas_estado2, equal_var=False)  

#     print(f"\nComparación entre {estado1} y {estado2}:")
#     print(f"T-Statistic: {t_stat:.4f}, P-Valor: {p_valor:.4f}")
    
#     if p_valor < alpha:
#         print(f"→ Se RECHAZA la hipótesis nula (H0). Las ventas son significativamente diferentes.")
#     else:
#         print(f"→ No hay suficiente evidencia para rechazar H0. Las ventas podrían ser iguales.")

# %%
pares_estados = [
    ('Washington', 'Arkansas'),
    ('Arkansas', 'Maryland'),
    ('Minnesota', 'Montana'),
    ('South Carolina', 'Connecticut')
]

alpha = 0.05

for estado1, estado2 in pares_estados:
    ventas_estado1 = df[df['State'] == estado1]['Sales']
    ventas_estado2 = df[df['State'] == estado2]['Sales']

    n1 = len(ventas_estado1)
    n2 = len(ventas_estado2)
    mu1 = ventas_estado1.mean()
    mu2 = ventas_estado2.mean()
    s1 = ventas_estado1.std(ddof=1)
    s2 = ventas_estado2.std(ddof=1)
    
    print(f"{estado1}: n={n1}, Media={mu1:.2f}, Desv={s1:.2f}")
    print(f"{estado2}: n={n2}, Media={mu2:.2f}, Desv={s2:.2f}")
    print('-'*30)

    gl = n1 + n2 - 2 
    
    s_pooled = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / gl)
    t_stat = (mu1 - mu2) / (s_pooled * np.sqrt(1/n1 + 1/n2))
    t_critico = stats.t.ppf(1 - alpha/2, gl)
    
    print(f"t_estadístico = {t_stat:.4f}, t_crítico = {t_critico:.4f}")
    
    if np.abs(t_stat) > t_critico:
        print("→ Rechazamos H0: Medias diferentes.\n")
    else:
        print("→ No hay evidencia para rechazar H0.\n")

# %% [markdown]
# ## 6. Verifica si estadísticamente las siguientes subcategorías tienen las mismas ventas: 
# -   (a) Accessories y Phones. 
# -   (b) Art y Envelopes. 
# -   (c) Paper y Storage.

# %%
# pares_subcategorias=[
#     ('Accessories','Phones'),
#     ('Art','Envelopes'),
#     ('Paper','Storage')
# ]
# alpha = 0.05
# for sub_cat_1, sub_cat_2 in pares_subcategorias:
#     ventas_sub_cat_1 = df[df['Sub-Category'] == sub_cat_1]['Sales']
#     ventas_sub_cat_2 = df[df['Sub-Category'] == sub_cat_2]['Sales']
#     t_stat, p_valor = stats.ttest_ind(ventas_sub_cat_1, ventas_sub_cat_2, equal_var=False)
#     print(f"\nComparación entre {sub_cat_1} y {sub_cat_2}:")
#     print(f"T-Statistic: {t_stat:.4f}, P-Valor: {p_valor:.4f}")
    
#     if p_valor < alpha:
#         print(f"→ Se RECHAZA la hipótesis nula (H0). Las ventas son significativamente diferentes.")
#     else:
#         print(f"→ No hay suficiente evidencia para rechazar H0. Las ventas podrían ser iguales.")

# %%
pares_subcategorias=[
    ('Accessories','Phones'),
    ('Art','Envelopes'),
    ('Paper','Storage')
]

alpha = 0.05

for sub_cat_1, sub_cat_2 in pares_subcategorias:
    ventas_sub_cat_1 = df[df['Sub-Category'] == sub_cat_1]['Sales']
    ventas_sub_cat_2 = df[df['Sub-Category'] == sub_cat_2]['Sales']

    n1 = len(ventas_sub_cat_1)
    n2 = len(ventas_sub_cat_2)
    mu1 = ventas_sub_cat_1.mean()
    mu2 = ventas_sub_cat_2.mean()
    s1 = ventas_sub_cat_1.std(ddof=1)
    s2 = ventas_sub_cat_2.std(ddof=1)
    
    print(f"{sub_cat_1}: n={n1}, Media={mu1:.2f}, Desv={s1:.2f}")
    print(f"{sub_cat_2}: n={n2}, Media={mu2:.2f}, Desv={s2:.2f}")
    print('-'*30)

    gl = n1 + n2 - 2 
    
    s_pooled = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / gl)
    t_stat = (mu1 - mu2) / (s_pooled * np.sqrt(1/n1 + 1/n2))
    t_critico = stats.t.ppf(1 - alpha/2, gl)
    
    print(f"t_estadístico = {t_stat:.4f}, t_crítico = {t_critico:.4f}")
    
    if np.abs(t_stat) > t_critico:
        print("→ Rechazamos H0: Medias diferentes.\n")
    else:
        print("→ No hay evidencia para rechazar H0.\n")

# %% [markdown]
# ## 7. En función de lo observado hacer comparaciones adicionales que considere pertinentes.

# %%
df['Category'].unique()

# %%
# pares_categorias=[
#     ('Office Supplies','Technology'),
#     ('Technology','Furniture'),
#     ('Furniture','Office Supplies')
# ]
# alpha = 0.05
# for cat_1, cat_2 in pares_subcategorias:
#     ventas_cat_1 = df[df['Sub-Category'] == cat_1]['Sales']
#     ventas_cat_2 = df[df['Sub-Category'] == cat_2]['Sales']
#     t_stat, p_valor = stats.ttest_ind(ventas_cat_1, ventas_cat_2, equal_var=False)
#     print(f"\nComparación entre {cat_1} y {cat_2}:")
#     print(f"T-Statistic: {t_stat:.4f}, P-Valor: {p_valor:.4f}")
    
#     if p_valor < alpha:
#         print(f"→ Se RECHAZA la hipótesis nula (H0). Las ventas son significativamente diferentes.")
#     else:
#         print(f"→ No hay suficiente evidencia para rechazar H0. Las ventas podrían ser iguales.")

# %%
pares_categorias=[
    ('Office Supplies','Technology'),
    ('Technology','Furniture'),
    ('Furniture','Office Supplies')
]

alpha = 0.05

for cat_1, cat_2 in pares_categorias:
    ventas_cat_1 = df[df['Category'] == cat_1]['Sales']
    ventas_cat_2 = df[df['Category'] == cat_2]['Sales']

    n1 = len(ventas_cat_1)
    n2 = len(ventas_cat_2)
    mu1 = ventas_cat_1.mean()
    mu2 = ventas_cat_2.mean()
    s1 = ventas_cat_1.std(ddof=1)
    s2 = ventas_cat_2.std(ddof=1)
    
    print(f"{cat_1}: n={n1}, Media={mu1:.2f}, Desv={s1:.2f}")
    print(f"{cat_2}: n={n2}, Media={mu2:.2f}, Desv={s2:.2f}")
    print('-'*30)

    gl = n1 + n2 - 2 
    
    s_pooled = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / gl)
    t_stat = (mu1 - mu2) / (s_pooled * np.sqrt(1/n1 + 1/n2))
    t_critico = stats.t.ppf(1 - alpha/2, gl)
    
    print(f"t_estadístico = {t_stat:.4f}, t_crítico = {t_critico:.4f}")
    
    if np.abs(t_stat) > t_critico:
        print("→ Rechazamos H0: Medias diferentes.\n")
    else:
        print("→ No hay evidencia para rechazar H0.\n")

# %% [markdown]
# ## 8. Da recomendaciones en función de lo que analizó para poner mayor énfasis de marketing en las áreas donde tenemos mejores ventas 

# %% [markdown]
# 1. Estados:
# *   No hay diferencias significativas en ventas entre los pares analizados (ej: Washington vs Arkansas).
# 
# *   Acción:
# 
# *       Mantén campañas generales en estos estados.
# 
# *       Monitorea nichos locales o eventos en estados con muestras pequeñas (ej: South Carolina, Montana).
# 
# 2. Subcategorías de Productos:
# *   Enfócate en:
# 
# *       Technology (Media ≈ 477.57), Phones (≈354.29), Furniture (≈356.75) y Storage (≈235.10), que tienen ventas significativamente altas.
# 
# *   Revisa estrategias en:
# 
# *       Office Supplies, Art y Paper, cuyas ventas son bajas. Considera reasignar recursos si no son estratégicos.
# 
# 3. Recomendación Integral:
# *   **Prioriza**: Campañas segmentadas para Technology, Phones, Furniture y Storage con promociones y alianzas estratégicas.
# 
# *   **Mantén vigilancia** en estados con potencial oculto (ej: Minnesota, Washington) y ajusta según tendencias emergentes.


