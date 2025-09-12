# T2 – Exploración de Datos (EDA)

# T2 – Exploración de Datos (EDA)

**Propósito.** Entender la distribución y relaciones principales del dataset limpio para orientar el modelamiento.

## Input
- `../T3_preparacion/incidents_clean_for_model.csv` (generado en T3)

## Cómo ejecutar
1) Abrir `EDA.ipynb`.  
2) Ejecutar todas las celdas.  
3) Las figuras se guardan en `./figs/` y el resumen en `resumen_eda.csv`.

## Qué incluye
- `info()` y `describe()` de columnas numéricas
- % de nulos por columna
- Distribución de `ttr_h` (y variantes: `ttr_h_log`, `ttr_h_winsor` si existen)
- Box/violin por categorías (p. ej. `priority_level`, `category`, `contact_type`)
- Serie mensual y heatmap día–hora si existe `opened_at`
- Top 10 de categorías

## Hallazgos (completar en bullets)
-Distribución del objetivo (ttr_h)

-ttr_h es altamente asimétrico a la derecha con colas largas (casos > 3.000–8.000 h).

-La transformación log1p reduce bastante la asimetría y la winsorización estabiliza los extremos sin perder señal.

-El % de nulos es el mismo en ttr_h, ttr_h_log, ttr_h_winsor y en los bines, lo cual es consistente: cuando falta ttr_h (ticket aún abierto), faltan sus derivados.

Relación con prioridad (priority_level)

-Las medianas de 1 y 2 son parecidas, pero prioridad 3 concentra la cola más pesada (muchos casos muy lentos).

Conclusión: la prioridad es variable explicativa relevante; baja prioridad tiende a acumular tiempos de resolución altos.

Patrones temporales (día/hora)

--Aperturas concentradas en horario laboral (≈ 8–17 h), con un pico fuerte los lunes 9–11 h.

Fines de semana son bajos.

Implicación: conviene crear features de calendario (dow, hour, is_weekend, month) para modelado y tablero.

Serie mensual de aperturas

-Muy alto volumen entre mar–may 2016 y luego cae casi a cero desde jun 2016.

-Esto no es típico de operación continua → ver “posible inconsistencia” abajo.

Categorías

-A nivel subcategory la gráfica es difícil de leer por alta cardinalidad; hay muchas clases raras.

-Para análisis/modelo usaré category o top-N subcategorías y agrupar el resto en other. (Para T4: considerar target encoding con validación estricta).

Nulos y duplicados

-Los nulos principales están en el objetivo (tickets no resueltos).

Mantengo un flag de abierto/no resuelto y NO elimino outliers: quedan marcados en ttr_outlier y también genero ttr_h_winsor para modelos robustos (alineado con lo pedido en clase).

## Notas
- La **limpieza** (faltantes, duplicados, outliers) se hace en **T3**.
